"""
MAS Core - LLM Integration (v1.2.0 Enhanced)
增强版 LLM 客户端，支持：
- 智能模型识别（解析动态反射）
- 代码片段分析
- LLM 驱动的 report 插入
- 寻优空间张开
"""
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
import openai


class LLMClient:
    """
    增强版 LLM 客户端 - v1.2.0
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.models = ["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"]
        self.current_model = None
    
    def _call_llm(self, prompt: str, system_msg: str = "", temperature: float = 0.2) -> str:
        """调用真实 LLM"""
        last_error = None
        
        for model in self.models:
            try:
                messages = []
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                
                self.current_model = model
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                continue
        
        raise Exception(f"All LLM models failed. Last error: {last_error}")
    
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
        使用真实 LLM 分析代码，识别 NAS 候选参数
        """
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
        
        except Exception as e:
            print(f"[LLMClient] Error analyzing code: {e}")
            return []
    
    def analyze_code_snippet_for_model_instantiation(self, code_snippet: str, 
                                                      available_models: List[str]) -> Dict[str, Any]:
        """
        分析代码片段，识别实际被实例化的模型
        
        Args:
            code_snippet: 代码片段（包含 model 实例化相关代码）
            available_models: 可用的模型类名列表
            
        Returns:
            Dict: {
                "instantiated_model": "模型类名",
                "instantiation_line": 行号,
                "model_variable": "模型变量名",
                "confidence": "high/medium/low"
            }
        """
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
        except Exception as e:
            print(f"[LLMClient] Error analyzing model instantiation: {e}")
            return {}
    
    def find_training_function_and_metrics(self, code_snippet: str) -> Dict[str, Any]:
        """
        分析代码片段，找到训练函数和 metrics
        
        Args:
            code_snippet: 训练相关的代码片段
            
        Returns:
            Dict: {
                "training_function": "函数名或类名",
                "function_type": "function/class",
                "metrics": {
                    "loss": "loss变量名",
                    "accuracy": "accuracy变量名",
                    ...
                },
                "model_variable": "模型变量名",
                "insertion_point": "插入位置描述"
            }
        """
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
        except Exception as e:
            print(f"[LLMClient] Error finding training function: {e}")
            return {}
    
    def analyze_conditional_layers(self, code_snippet: str) -> List[Dict[str, Any]]:
        """
        分析条件层代码（如 if activation == 'relu': self.act = nn.ReLU()）
        识别可以转换为 LayerSpace 的条件分支
        
        Args:
            code_snippet: 包含条件层选择的代码片段
            
        Returns:
            List[Dict]: 每个条件层的信息
            [{
                "variable_name": "self.act",
                "condition_variable": "activation",
                "options": ["nn.ReLU()", "nn.Sigmoid()", "nn.Tanh()"],
                "line_start": 10,
                "line_end": 20
            }]
        """
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
        except Exception as e:
            print(f"[LLMClient] Error analyzing conditional layers: {e}")
            return []
    
    def generate_layer_space_replacement(self, variable_name: str, 
                                          options: List[str]) -> str:
        """
        生成 LayerSpace 替换代码
        
        Args:
            variable_name: 变量名
            options: 层选项列表
            
        Returns:
            str: LayerSpace 代码字符串
        """
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
        except Exception as e:
            return f"解析失败: {e}"
    
    def generate_search_space(self, param_name: str, current_value: Any, 
                              param_type: str) -> List[Any]:
        """使用真实 LLM 生成搜索空间"""
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
        except:
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
        except:
            return []


# 全局 LLM 客户端
_llm_client = None

def init_llm(api_key: str, base_url: str):
    """初始化 LLM 客户端"""
    global _llm_client
    _llm_client = LLMClient(api_key=api_key, base_url=base_url)

def get_llm_client() -> LLMClient:
    """获取 LLM 客户端"""
    global _llm_client
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized. Call init_llm() first.")
    return _llm_client
