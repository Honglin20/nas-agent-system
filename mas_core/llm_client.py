"""
MAS Core - LLM Integration (v1.4.0 Fixed)
修复版 LLM 客户端，改进：
- 参数过滤：只推荐模型超参和结构参数
- 不推荐的参数：lr、optimizer、num_classes、batch_size、epoch 等训练参数
- 推荐的参数：d_model、num_layers、num_heads、dropout、dim_feedforward、activation、norm_type 等
- 支持 YAML 配置文件分析
- 智能模型修改范围限制
"""
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

import openai
import httpx

from .exceptions import (
    LLMError, ErrorCode, 
    raise_llm_timeout, raise_llm_api_error
)
from .retry_cache import retry_with_backoff, RetryConfig, CircuitBreaker
from .config import get_config


# v1.4.0: 参数分类定义
# 不推荐的参数（训练相关、任务相关）
EXCLUDED_PARAM_NAMES = {
    # 学习率相关
    'lr', 'learning_rate', 'learning-rate', 'learningrate',
    # 优化器相关
    'optimizer', 'optim',
    # 任务相关（固定）
    'num_classes', 'num_classes', 'n_classes', 'nclasses', 'output_dim', 'output_dim',
    'input_dim', 'input_dim', 'in_features', 'out_features',
    # 批次相关
    'batch_size', 'batchsize', 'batch-size',
    # 训练轮数
    'epoch', 'epochs', 'num_epochs', 'num_epochs', 'max_epochs',
    # 其他训练参数
    'weight_decay', 'weight_decay', 'momentum', 'beta1', 'beta2',
    'warmup_steps', 'warmup_epochs', 'grad_clip', 'max_grad_norm',
    # 数据相关
    'num_samples', 'dataset_size', 'train_size', 'val_size', 'test_size',
}

EXCLUDED_PARAM_PATTERNS = [
    r'.*lr$', r'.*learning[_-]?rate$',
    r'.*optimizer.*',
    r'.*batch[_-]?size$',
    r'.*epoch.*',
    r'.*num[_-]?classes$',
    r'.*n[_-]?classes$',
    r'.*weight[_-]?decay$',
]

# 推荐的参数（模型结构和超参）
RECOMMENDED_PARAM_NAMES = {
    # 模型维度
    'd_model', 'dmodel', 'hidden_dim', 'hidden_dim', 'embed_dim', 'embedding_dim',
    'hidden_size', 'hidden', 'dim', 'n_dim', 'feature_dim',
    # 模型深度
    'num_layers', 'num_layers', 'n_layers', 'nlayers', 'depth', 'n_layer',
    'num_blocks', 'num_blocks', 'n_blocks',
    # 注意力头
    'num_heads', 'num_heads', 'n_heads', 'nheads', 'nhead', 'num_attention_heads',
    # Dropout / 正则化
    'dropout', 'dropout_rate', 'attention_dropout', 'attn_dropout',
    'dropout_prob', 'p_dropout', 'hidden_dropout_prob',
    # 前馈维度
    'dim_feedforward', 'dim_feedforward', 'ffn_dim', 'ff_dim', 'intermediate_size',
    'hidden_dim_ff', 'feedforward_dim',
    # 激活函数
    'activation', 'activation', 'act', 'hidden_act',
    # 归一化
    'norm_type', 'norm_type', 'norm', 'layer_norm_eps', 'normalization',
    # 其他结构参数
    'max_len', 'max_length', 'max_position_embeddings', 'seq_length',
    'num_attention_heads', 'vocab_size', 'type_vocab_size',
    'kernel_size', 'kernel_sizes', 'filter_size', 'num_filters',
    'pool_size', 'stride', 'padding',
}

RECOMMENDED_PARAM_PATTERNS = [
    r'.*d[_-]?model.*',
    r'.*hidden[_-]?dim.*',
    r'.*embed[_-]?dim.*',
    r'.*num[_-]?layers.*',
    r'.*n[_-]?layers.*',
    r'.*num[_-]?heads.*',
    r'.*n[_-]?heads.*',
    r'.*dropout.*',
    r'.*feedforward.*',
    r'.*ffn[_-]?dim.*',
    r'.*activation.*',
    r'.*norm[_-]?type.*',
]


def is_excluded_param(param_name: str) -> bool:
    """检查参数是否应该被排除（不推荐寻优）"""
    param_lower = param_name.lower().strip()
    
    # 直接匹配排除列表
    if param_lower in EXCLUDED_PARAM_NAMES:
        return True
    
    # 正则匹配排除模式
    for pattern in EXCLUDED_PARAM_PATTERNS:
        if re.match(pattern, param_lower, re.IGNORECASE):
            return True
    
    return False


def is_recommended_param(param_name: str) -> bool:
    """检查参数是否是推荐的模型结构参数"""
    param_lower = param_name.lower().strip()
    
    # 直接匹配推荐列表
    if param_lower in RECOMMENDED_PARAM_NAMES:
        return True
    
    # 正则匹配推荐模式
    for pattern in RECOMMENDED_PARAM_PATTERNS:
        if re.match(pattern, param_lower, re.IGNORECASE):
            return True
    
    return False


def filter_nas_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    v1.4.0: 过滤 NAS 候选参数，只保留模型结构参数
    
    Args:
        candidates: 原始候选参数列表
        
    Returns:
        过滤后的候选参数列表
    """
    filtered = []
    
    for cand in candidates:
        param_name = cand.get('name', '')
        
        # 首先检查是否在排除列表中
        if is_excluded_param(param_name):
            print(f"[LLMClient] Excluding training/task parameter: {param_name}")
            continue
        
        # 检查是否是推荐的模型参数
        if is_recommended_param(param_name):
            filtered.append(cand)
            print(f"[LLMClient] Including model parameter: {param_name}")
        else:
            # 对于不在明确列表中的参数，保留但标记为待定
            # 让 LLM 进一步判断
            cand['_pending_review'] = True
            filtered.append(cand)
            print(f"[LLMClient] Pending review for parameter: {param_name}")
    
    return filtered


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""
    
    @abstractmethod
    def analyze_code_for_nas(self, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """分析代码，识别 NAS 候选参数"""
        pass
    
    @abstractmethod
    def generate_search_space(self, param_name: str, current_value: Any, 
                              param_type: str) -> List[Any]:
        """生成搜索空间"""
        pass
    
    @abstractmethod
    def recommend_injection(self, candidates: List[Dict]) -> List[Dict]:
        """推荐注入参数"""
        pass


class LLMClient(BaseLLMClient):
    """
    修复版 LLM 客户端 - v1.4.0
    - 参数过滤：只推荐模型超参和结构参数
    - 支持 YAML 配置文件分析
    - 智能模型修改范围限制
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        config = get_config()
        
        self.api_key = api_key or config.llm.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or config.llm.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # v1.4.0: 检查 API key 和 URL
        if not self.api_key:
            raise LLMError(
                ErrorCode.LLM_NOT_INITIALIZED,
                "LLM API Key 未设置。请通过以下方式之一配置:\n"
                "  1. 设置环境变量: export OPENAI_API_KEY='your-api-key'\n"
                "  2. 在 ~/.nas-cli/config.yaml 中配置 llm.api_key\n"
                "  3. 调用 init_llm(api_key='your-api-key')"
            )
        
        if not self.base_url:
            raise LLMError(
                ErrorCode.LLM_NOT_INITIALIZED,
                "LLM Base URL 未设置。请通过以下方式之一配置:\n"
                "  1. 设置环境变量: export OPENAI_BASE_URL='https://api.openai.com/v1'\n"
                "  2. 在 ~/.nas-cli/config.yaml 中配置 llm.base_url\n"
                "  3. 调用 init_llm(base_url='your-base-url')"
            )
        
        self.timeout = config.llm.timeout
        self.max_retries = config.llm.max_retries
        self.retry_delay = config.llm.retry_delay
        self.temperature = config.llm.temperature
        
        self.models = config.llm.models
        self.current_model = None
        
        # v1.4.0: 从环境变量获取代理设置
        self.http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        self.https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        
        # v1.4.0: 初始化 OpenAI 客户端，支持代理和禁用 SSL 验证
        http_client = None
        if self.http_proxy or self.https_proxy:
            proxies = {}
            if self.http_proxy:
                proxies['http://'] = self.http_proxy
            if self.https_proxy:
                proxies['https://'] = self.https_proxy
            
            # 创建支持代理的 httpx client
            http_client = httpx.Client(
                verify=False,
                proxies=proxies if proxies else None,
                timeout=self.timeout
            )
        else:
            # 无代理时仍禁用 SSL 验证以兼容某些环境
            http_client = httpx.Client(verify=False, timeout=self.timeout)
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client
        )
        
        # 熔断器
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # v1.4.0: 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """v1.4.0: 测试 LLM 连接是否可用"""
        try:
            # 发送一个简单的测试请求
            response = self.client.chat.completions.create(
                model=self.models[0] if self.models else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=10
            )
            print(f"[LLMClient] ✓ LLM 连接测试成功 (使用模型: {self.models[0]})")
        except openai.APITimeoutError as e:
            proxy_hint = ""
            if not self.http_proxy and not self.https_proxy:
                proxy_hint = (
                    "\n\n提示: 连接超时可能是网络问题。如果您在中国大陆，"
                    "可能需要配置代理:\n"
                    "  export http_proxy=http://127.0.0.1:7890\n"
                    "  export https_proxy=http://127.0.0.1:7890"
                )
            raise LLMError(
                ErrorCode.LLM_CONNECTION_ERROR,
                f"LLM 连接测试超时: {e}{proxy_hint}"
            )
        except openai.AuthenticationError as e:
            raise LLMError(
                ErrorCode.LLM_AUTHENTICATION_ERROR,
                f"LLM API Key 验证失败: {e}\n"
                "请检查您的 API Key 是否正确设置。"
            )
        except Exception as e:
            raise LLMError(
                ErrorCode.LLM_CONNECTION_ERROR,
                f"LLM 连接测试失败: {e}"
            )
    
    def _call_llm(self, prompt: str, system_msg: str = "", 
                  temperature: Optional[float] = None) -> str:
        """
        调用 LLM，带重试和熔断保护
        """
        # 检查熔断器
        if not self._circuit_breaker.can_execute():
            raise LLMError(
                ErrorCode.LLM_API_ERROR,
                "服务暂时不可用，请稍后重试（熔断器开启）"
            )
        
        temp = temperature or self.temperature
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            for model in self.models:
                try:
                    messages = []
                    if system_msg:
                        messages.append({"role": "system", "content": system_msg})
                    messages.append({"role": "user", "content": prompt})
                    
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temp,
                        timeout=self.timeout
                    )
                    
                    self.current_model = model
                    
                    # 记录成功
                    self._circuit_breaker.record_success()
                    
                    return response.choices[0].message.content
                    
                except openai.APITimeoutError as e:
                    last_error = e
                    print(f"[LLMClient] Timeout with model {model}: {e}")
                    # v1.4.0: 超时错误提示代理配置
                    if attempt == self.max_retries and not self.http_proxy and not self.https_proxy:
                        print(f"[LLMClient] 提示: 如果连接持续超时，请尝试配置代理:")
                        print(f"  export http_proxy=http://127.0.0.1:7890")
                        print(f"  export https_proxy=http://127.0.0.1:7890")
                    continue
                except openai.RateLimitError as e:
                    last_error = e
                    print(f"[LLMClient] Rate limit with model {model}: {e}")
                    # 速率限制时等待更长时间
                    time.sleep(self.retry_delay * 5)
                    continue
                except openai.APIError as e:
                    last_error = e
                    print(f"[LLMClient] API error with model {model}: {e}")
                    continue
                except Exception as e:
                    last_error = e
                    print(f"[LLMClient] Unexpected error with model {model}: {e}")
                    continue
            
            # 所有模型都失败，等待后重试
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # 指数退避
                print(f"[LLMClient] All models failed, retrying in {delay}s...")
                time.sleep(delay)
        
        # 所有重试都失败
        self._circuit_breaker.record_failure()
        
        raise LLMError(
            ErrorCode.LLM_ALL_MODELS_FAILED,
            f"所有 LLM 模型都调用失败，已重试 {self.max_retries} 次",
            details={'last_error': str(last_error)}
        )
    
    def _extract_json(self, content: str) -> Optional[Dict]:
        """从 LLM 响应中提取 JSON"""
        # 尝试1: 直接解析
        try:
            return json.loads(content)
        except:
            pass
        
        # 尝试2: 从代码块提取
        try:
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        # 尝试3: 从任意 JSON 对象提取
        try:
            match = re.search(r'(\{[\s\S]*"[\w_]+"[\s\S]*\})', content)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        return None
    
    def analyze_code_for_nas(self, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """
        v1.4.0: 使用真实 LLM 分析代码，识别 NAS 候选参数
        只推荐模型超参和结构参数，过滤掉训练参数
        """
        # 检查代码长度
        if len(code) > 50000:  # 约 50KB
            print(f"[LLMClient] Warning: File {file_path} is too large, truncating...")
            code = code[:50000] + "\n# ... (truncated)"
        
        prompt = f"""你是一个神经网络架构搜索（NAS）专家。请分析以下 Python 代码，识别所有可以寻优的**模型结构参数**。

文件路径: {file_path}

代码:
```python
{code}
```

请仔细分析代码，只识别以下类型的**模型结构参数**：
1. 模型维度参数（如 d_model、hidden_dim、embed_dim、dim_feedforward 等）
2. 模型深度参数（如 num_layers、n_layers、depth、num_blocks 等）
3. 注意力头数（如 num_heads、n_heads、nhead 等）
4. Dropout 率（如 dropout、attention_dropout 等正则化参数）
5. 激活函数选择（如 activation、hidden_act 等）
6. 归一化类型（如 norm_type、normalization 等）
7. 其他架构参数（如 kernel_size、max_len 等）

**重要：不要推荐以下训练相关参数：**
- learning_rate / lr（学习率）
- optimizer（优化器）
- batch_size（批次大小）
- num_epochs / epochs（训练轮数）
- weight_decay（权重衰减）
- momentum、beta1、beta2（优化器参数）
- num_classes / num_classes（类别数，任务相关）

对于每个参数，请提供:
- name: 参数名
- type: "value" (数值) 或 "layer" (层选择)
- current_value: 当前值
- suggested_space: 建议的搜索空间（用Python表达式表示）
- reason: 为什么这个参数值得寻优
- line: 大概的行号（如果能在代码中定位）
- is_backbone_param: true/false（是否是 backbone 主模型的参数）

必须以 JSON 格式返回，格式如下:
{{"candidates": [{{"name": "d_model", "type": "value", "current_value": "256", "suggested_space": "ValueSpace([128, 256, 512])", "reason": "模型维度影响表达能力", "line": 15, "is_backbone_param": true}}]}}

重要：返回的 JSON 必须在一行内，不要格式化换行。"""
        
        try:
            content = self._call_llm(
                prompt,
                "你是一个专业的神经网络架构搜索（NAS）专家，擅长识别深度学习代码中的可寻优模型结构参数。"
            )
            
            result = self._extract_json(content)
            if result and "candidates" in result:
                candidates = result["candidates"]
            elif isinstance(result, list):
                candidates = result
            else:
                candidates = []
            
            # v1.4.0: 应用参数过滤
            print(f"[LLMClient] LLM returned {len(candidates)} candidates, applying filter...")
            filtered_candidates = filter_nas_candidates(candidates)
            print(f"[LLMClient] After filtering: {len(filtered_candidates)} candidates")
            
            return filtered_candidates
        
        except LLMError as e:
            print(f"[LLMClient] Error analyzing code: {e}")
            return []
    
    def analyze_yaml_config_for_nas(self, yaml_content: str, file_path: str = "") -> List[Dict[str, Any]]:
        """
        v1.4.0: 分析 YAML 配置文件，识别可寻优参数
        
        Args:
            yaml_content: YAML 文件内容
            file_path: 文件路径
            
        Returns:
            候选参数列表
        """
        prompt = f"""你是一个神经网络架构搜索（NAS）专家。请分析以下 YAML 配置文件，识别所有可以寻优的**模型结构参数**。

文件路径: {file_path}

YAML 内容:
```yaml
{yaml_content}
```

请提取所有模型结构参数及其当前值，并推荐搜索空间。

**只关注模型结构参数，不要推荐训练参数（如 lr、batch_size、epochs、optimizer 等）。**

推荐的参数类型：
- 模型维度：d_model、hidden_dim、embed_dim、dim_feedforward 等
- 模型深度：num_layers、n_layers、depth 等
- 注意力头数：num_heads、n_heads 等
- Dropout：dropout、attention_dropout 等
- 激活函数：activation 等
- 归一化：norm_type 等

对于每个参数，请提供:
- name: 参数名（完整路径，如 model.d_model）
- type: "value" 或 "layer"
- current_value: 当前值
- suggested_space: 建议搜索空间
- reason: 推荐理由
- yaml_path: YAML 中的路径（如 ['model', 'd_model']）

以 JSON 格式返回: {{"candidates": [...]}}"""

        try:
            content = self._call_llm(
                prompt,
                "你是一个专业的 NAS 专家，擅长分析配置文件中的模型结构参数。"
            )
            
            result = self._extract_json(content)
            if result and "candidates" in result:
                candidates = result["candidates"]
            elif isinstance(result, list):
                candidates = result
            else:
                candidates = []
            
            # 应用参数过滤
            filtered_candidates = filter_nas_candidates(candidates)
            return filtered_candidates
            
        except LLMError as e:
            print(f"[LLMClient] Error analyzing YAML config: {e}")
            return []
    
    def identify_backbone_init_params(self, code: str, model_class_name: str) -> Dict[str, Any]:
        """
        v1.4.0: 识别 backbone 主模型 __init__ 方法的参数
        
        Args:
            code: 模型类代码
            model_class_name: 模型类名
            
        Returns:
            参数信息字典
        """
        prompt = f"""你是一个 Python 代码分析专家。请分析以下模型类代码，识别 backbone 主模型的 __init__ 方法参数。

模型类名: {model_class_name}

代码:
```python
{code}
```

请分析:
1. __init__ 方法有哪些参数？
2. 每个参数的当前默认值是什么？
3. 哪些参数是模型结构参数（值得 NAS 寻优）？
4. 哪些参数是训练参数（不应该寻优）？

以 JSON 格式返回:
{{
    "init_params": [
        {{"name": "d_model", "default_value": "256", "is_structure_param": true, "is_training_param": false}}
    ],
    "backbone_class": "类名",
    "line_start": 10,
    "line_end": 50
}}"""

        try:
            content = self._call_llm(
                prompt,
                "你是一个 Python 代码分析专家，擅长分析深度学习模型类的结构。"
            )
            
            result = self._extract_json(content)
            return result or {}
            
        except LLMError as e:
            print(f"[LLMClient] Error identifying backbone params: {e}")
            return {}
    
    def analyze_code_snippet_for_model_instantiation(self, code_snippet: str, 
                                                      available_models: List[str]) -> Dict[str, Any]:
        """分析代码片段，识别实际被实例化的模型"""
        prompt = f"""你是一个 Python 代码分析专家。请分析以下代码片段，识别实际被实例化的模型类。

可用模型类: {available_models}

代码片段:
```python
{code_snippet}
```

请分析:
1. 哪个模型类被实际实例化了？（通过 getattr 或直接调用）
2. 实例化发生在哪一行？
3. 模型被赋值给哪个变量？
4. 你的置信度如何？

必须以 JSON 格式返回:
{{
    "instantiated_model": "模型类名",
    "instantiation_line": 行号,
    "model_variable": "模型变量名",
    "confidence": "high/medium/low",
    "reasoning": "分析理由"
}}

只返回 JSON，不要其他内容。"""
        
        try:
            content = self._call_llm(prompt, "你是一个 Python 代码分析专家，擅长解析动态反射和模型实例化。")
            result = self._extract_json(content)
            return result or {}
        except LLMError as e:
            print(f"[LLMClient] Error analyzing model instantiation: {e}")
            return {}
    
    def find_training_function_and_metrics(self, code_snippet: str) -> Dict[str, Any]:
        """分析代码片段，找到训练函数和 metrics"""
        prompt = f"""你是一个深度学习代码分析专家。请分析以下代码片段，找到训练函数/类和关键指标。

代码片段:
```python
{code_snippet}
```

请分析:
1. 训练函数或类是什么？
2. 有哪些关键指标（loss, accuracy, reward 等）？
3. 模型变量名是什么？
4. 在哪里应该插入 report(model=model, ...) 调用？

必须以 JSON 格式返回:
{{
    "training_function": "函数名或类名",
    "function_type": "function/class",
    "metrics": {{
        "loss": "loss变量名",
        "accuracy": "accuracy变量名"
    }},
    "model_variable": "模型变量名",
    "insertion_point": "每个 epoch 结束后，在打印日志之前"
}}

只返回 JSON，不要其他内容。"""
        
        try:
            content = self._call_llm(prompt, "你是一个深度学习代码分析专家，擅长识别训练循环和指标。")
            result = self._extract_json(content)
            return result or {}
        except LLMError as e:
            print(f"[LLMClient] Error finding training function: {e}")
            return {}
    
    def analyze_conditional_layers(self, code_snippet: str) -> List[Dict[str, Any]]:
        """分析条件层代码"""
        prompt = f"""你是一个神经网络架构搜索专家。请分析以下代码片段，识别可以转换为 LayerSpace 的条件层选择。

代码片段:
```python
{code_snippet}
```

请识别所有类似以下的模式:
- if activation == 'relu': self.act = nn.ReLU()
- if norm_type == 'batchnorm': self.norm = nn.BatchNorm1d(...)
- 等等

对于每个条件层选择，提供:
- variable_name: 被赋值的变量名
- condition_variable: 条件判断的变量名
- options: 所有可能的选项列表
- line_start: 开始行号
- line_end: 结束行号

必须以 JSON 格式返回:
{{
    "conditional_layers": [
        {{
            "variable_name": "self.act",
            "condition_variable": "activation",
            "options": ["nn.ReLU()", "nn.Sigmoid()", "nn.Tanh()"],
            "line_start": 10,
            "line_end": 16
        }}
    ]
}}

只返回 JSON，不要其他内容。"""
        
        try:
            content = self._call_llm(prompt, "你是一个神经网络架构搜索专家，擅长识别条件层选择模式。")
            result = self._extract_json(content)
            return result.get("conditional_layers", []) if result else []
        except LLMError as e:
            print(f"[LLMClient] Error analyzing conditional layers: {e}")
            return []
    
    def generate_layer_space_replacement(self, variable_name: str, 
                                          options: List[str]) -> str:
        """生成 LayerSpace 替换代码"""
        formatted_options = [f'"{opt}"' if not opt.startswith('nn.') else opt for opt in options]
        return f"LayerSpace([{', '.join(formatted_options)}])"
    
    def resolve_dynamic_reference(self, code: str, variable_name: str) -> str:
        """使用真实 LLM 解析动态引用（如 getattr）"""
        prompt = f"""请分析以下 Python 代码，解析动态引用的实际值。

代码:
```python
{code}
```

需要解析的变量: {variable_name}

请分析:
1. 这个变量的实际值是什么？
2. 它的类型是什么？
3. 如果是动态加载的（如 getattr），实际会加载什么？

请详细说明你的推理过程。"""
        
        try:
            return self._call_llm(prompt, "你是一个 Python 代码分析专家，擅长解析动态引用和反射。")
        except LLMError as e:
            return f"解析失败: {e}"
    
    def generate_search_space(self, param_name: str, current_value: Any, 
                              param_type: str) -> List[Any]:
        """使用真实 LLM 生成搜索空间"""
        # 首先检查是否有预定义的搜索空间
        config = get_config()
        for key, space in config.nas.default_search_space_sizes.items():
            if key in param_name.lower():
                return space
        
        prompt = f"""你是一个 NAS 专家。请为以下参数生成合适的搜索空间。

参数名: {param_name}
当前值: {current_value}
类型: {param_type}

请生成一个合理的搜索空间列表，返回 JSON 格式:
{{
    "search_space": [值1, 值2, 值3, ...]
}}

例如:
- 学习率: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- 维度: [32, 64, 128, 256]
- 层数: [1, 2, 3, 4]

只返回 JSON。"""
        
        try:
            content = self._call_llm(prompt)
            result = self._extract_json(content)
            return result.get("search_space", [current_value]) if result else [current_value]
        except LLMError:
            # 降级到默认逻辑
            if isinstance(current_value, (int, float)):
                if 'lr' in param_name.lower() or 'rate' in param_name.lower():
                    if current_value < 1:
                        return [current_value / 10, current_value, current_value * 10]
                return [max(1, int(current_value / 2)), current_value, current_value * 2]
            return [current_value]
    
    def recommend_injection(self, candidates: List[Dict]) -> List[Dict]:
        """使用真实 LLM 推荐哪些参数值得注入"""
        # v1.4.0: 先过滤掉训练参数
        candidates = filter_nas_candidates(candidates)
        
        if not candidates:
            return []
        
        candidates_str = json.dumps(candidates, indent=2, ensure_ascii=False)
        
        prompt = f"""你是一个 NAS 专家。请分析以下候选参数，推荐哪些最值得进行 NAS 寻优。

候选参数:
{candidates_str}

请对每个参数给出:
- recommended: true/false（是否推荐寻优）
- priority: "high"/"medium"/"low"（优先级）

**重要：只推荐模型结构参数（如 d_model、num_layers、num_heads、dropout、activation 等）。**
**不要推荐训练参数（如 lr、batch_size、epochs、optimizer 等）。**

返回 JSON 格式:
{{
    "recommendations": [
        {{
            "name": "参数名",
            "recommended": true,
            "priority": "high",
            "reason": "推荐理由"
        }}
    ]
}}

只返回 JSON。"""
        
        try:
            content = self._call_llm(prompt)
            result = self._extract_json(content)
            return result.get("recommendations", []) if result else []
        except LLMError:
            # 降级：全部推荐
            return [
                {
                    'name': c['name'],
                    'recommended': True,
                    'priority': 'medium',
                    'reason': 'Default recommendation (LLM unavailable)'
                }
                for c in candidates
            ]


# 全局 LLM 客户端
_llm_client = None

def init_llm(api_key: str = None, base_url: str = None, use_mock: bool = False):
    """初始化 LLM 客户端"""
    global _llm_client
    
    # v1.4.0: 移除 Mock 支持，强制使用真实 LLM
    if use_mock:
        print("[LLMClient] 警告: Mock 模式已在 v1.4.0 中移除，将使用真实 LLM")
    
    _llm_client = LLMClient(api_key=api_key, base_url=base_url)

def get_llm_client() -> BaseLLMClient:
    """获取 LLM 客户端"""
    global _llm_client
    if _llm_client is None:
        # 尝试自动初始化
        config = get_config()
        if config.llm.api_key:
            init_llm()
        else:
            raise LLMError(
                ErrorCode.LLM_NOT_INITIALIZED,
                "LLM 客户端未初始化。请设置 OPENAI_API_KEY 环境变量或调用 init_llm()"
            )
    return _llm_client

def is_llm_available() -> bool:
    """检查 LLM 是否可用"""
    try:
        client = get_llm_client()
        return client is not None
    except:
        return False
