"""
MAS Core - LLM Integration (v1.3.0 Enhanced)
增强版 LLM 客户端，支持：
- 智能模型识别（解析动态反射）
- 代码片段分析
- LLM 驱动的 report 插入
- 寻优空间张开
- 重试机制和超时控制
- 响应缓存
"""
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

import openai

from .exceptions import (
    LLMError, ErrorCode, 
    raise_llm_timeout, raise_llm_api_error
)
from .retry_cache import retry_with_backoff, RetryConfig, CircuitBreaker
from .config import get_config


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


class MockLLMClient(BaseLLMClient):
    """Mock LLM 客户端，用于测试"""
    
    def analyze_code_for_nas(self, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """模拟分析代码"""
        # 简单的规则匹配
        candidates = []
        
        # 查找常见的 NAS 参数模式
        patterns = [
            (r'(\w*[Ll]earning_[Rr]ate\w*|\w*lr\w*)\s*=\s*([0-9.]+)', 'learning_rate', 'value'),
            (r'(\w*[Bb]atch_[Ss]ize\w*)\s*=\s*(\d+)', 'batch_size', 'value'),
            (r'(\w*[Dd]ropout\w*)\s*=\s*([0-9.]+)', 'dropout', 'value'),
            (r'(\w*[Nn]um_[Ll]ayers\w*)\s*=\s*(\d+)', 'num_layers', 'value'),
            (r'(\w*[Hh]idden_[Dd]im\w*)\s*=\s*(\d+)', 'hidden_dim', 'value'),
        ]
        
        for pattern, name, param_type in patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                value = match.group(2)
                try:
                    value = float(value) if '.' in value else int(value)
                except:
                    pass
                
                candidates.append({
                    'name': name,
                    'type': param_type,
                    'current_value': value,
                    'suggested_space': f"ValueSpace([{value}])",
                    'reason': f'Detected {name} parameter',
                    'line': code[:match.start()].count('\n') + 1
                })
        
        return candidates
    
    def generate_search_space(self, param_name: str, current_value: Any, 
                              param_type: str) -> List[Any]:
        """模拟生成搜索空间"""
        if isinstance(current_value, (int, float)):
            if 'lr' in param_name.lower():
                return [current_value / 10, current_value, current_value * 10]
            return [max(1, int(current_value / 2)), current_value, current_value * 2]
        return [current_value]
    
    def recommend_injection(self, candidates: List[Dict]) -> List[Dict]:
        """模拟推荐"""
        return [
            {
                'name': c['name'],
                'recommended': True,
                'priority': 'medium',
                'reason': 'Mock recommendation'
            }
            for c in candidates
        ]
    
    def analyze_code_snippet_for_model_instantiation(self, code_snippet: str, 
                                                      available_models: List[str]) -> Dict[str, Any]:
        """模拟模型实例化分析"""
        return {
            "instantiated_model": available_models[0] if available_models else "Unknown",
            "instantiation_line": 1,
            "model_variable": "model",
            "confidence": "low",
            "reasoning": "Mock analysis"
        }
    
    def find_training_function_and_metrics(self, code_snippet: str) -> Dict[str, Any]:
        """模拟训练函数分析"""
        return {
            "training_function": "train",
            "function_type": "function",
            "metrics": {"loss": "loss", "accuracy": "acc"},
            "model_variable": "model",
            "insertion_point": "end of epoch"
        }
    
    def analyze_conditional_layers(self, code_snippet: str) -> List[Dict[str, Any]]:
        """模拟条件层分析"""
        return []
    
    def resolve_dynamic_reference(self, code: str, variable_name: str) -> str:
        """模拟动态引用解析"""
        return f"Mock resolution for {variable_name}"


class LLMClient(BaseLLMClient):
    """
    增强版 LLM 客户端 - v1.3.0
    - 支持重试机制和超时控制
    - 支持熔断器模式
    - 更好的错误处理
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        config = get_config()
        
        self.api_key = api_key or config.llm.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or config.llm.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.timeout = config.llm.timeout
        self.max_retries = config.llm.max_retries
        self.retry_delay = config.llm.retry_delay
        self.temperature = config.llm.temperature
        
        self.models = config.llm.models
        self.current_model = None
        
        # 初始化 OpenAI 客户端
        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        else:
            self.client = None
        
        # 熔断器
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
    
    def _call_llm(self, prompt: str, system_msg: str = "", 
                  temperature: Optional[float] = None) -> str:
        """
        调用 LLM，带重试和熔断保护
        """
        if not self.client:
            raise LLMError(
                ErrorCode.LLM_NOT_INITIALIZED,
                "LLM 客户端未初始化，请检查 API Key 配置"
            )
        
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
        """使用真实 LLM 分析代码，识别 NAS 候选参数"""
        # 检查代码长度
        if len(code) > 50000:  # 约 50KB
            print(f"[LLMClient] Warning: File {file_path} is too large, truncating...")
            code = code[:50000] + "\n# ... (truncated)"
        
        prompt = f"""你是一个神经网络架构搜索（NAS）专家。请分析以下 Python 代码，识别所有可以寻优的参数。

文件路径: {file_path}

代码:
```python
{code}
```

请仔细分析代码，识别以下类型的参数：
1. 数值参数（如学习率、维度、层数、批次大小、dropout率等）
2. 层选择（如激活函数、优化器、归一化层等）

对于每个参数，请提供:
- name: 参数名
- type: "value" (数值) 或 "layer" (层选择)
- current_value: 当前值
- suggested_space: 建议的搜索空间（用Python表达式表示）
- reason: 为什么这个参数值得寻优
- line: 大概的行号（如果能在代码中定位）

必须以 JSON 格式返回，格式如下:
{{"candidates": [{{"name": "learning_rate", "type": "value", "current_value": "0.001", "suggested_space": "ValueSpace([1e-4, 1e-3, 1e-2])", "reason": "学习率对模型收敛至关重要", "line": 15}}]}}

重要：返回的 JSON 必须在一行内，不要格式化换行。"""
        
        try:
            content = self._call_llm(
                prompt,
                "你是一个专业的神经网络架构搜索（NAS）专家，擅长识别深度学习代码中的可寻优参数。"
            )
            
            result = self._extract_json(content)
            if result and "candidates" in result:
                return result["candidates"]
            if isinstance(result, list):
                return result
            return []
        
        except LLMError as e:
            print(f"[LLMClient] Error analyzing code: {e}")
            return []
    
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
        candidates_str = json.dumps(candidates, indent=2, ensure_ascii=False)
        
        prompt = f"""你是一个 NAS 专家。请分析以下候选参数，推荐哪些最值得进行 NAS 寻优。

候选参数:
{candidates_str}

请对每个参数给出:
- recommended: true/false（是否推荐寻优）
- priority: "high"/"medium"/"low"（优先级）

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
    
    if use_mock:
        _llm_client = MockLLMClient()
    else:
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
