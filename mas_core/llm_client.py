"""
MAS Core - LLM Integration (Real LLM Only)
严禁使用规则模拟，所有分析必须通过真实 LLM
"""
import os
import re
import json
from typing import Dict, Any, List, Optional
import openai


class LLMClient:
    """
    LLM 客户端 - 所有分析必须通过真实 LLM
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 可用模型列表
        self.models = ["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"]
        self.current_model = None
    
    def _call_llm(self, prompt: str, system_msg: str = "") -> str:
        """
        调用真实 LLM
        """
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
                    temperature=0.2
                )
                
                self.current_model = model
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                continue
        
        raise Exception(f"All LLM models failed. Last error: {last_error}")
    
    def analyze_code_for_nas(self, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """
        使用真实 LLM 分析代码，识别 NAS 候选参数
        严禁使用规则模拟
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

必须以 JSON 格式返回，格式如下（注意：不要包含换行，保持在一行内）:
{{"candidates": [{{"name": "learning_rate", "type": "value", "current_value": "0.001", "suggested_space": "ValueSpace([1e-4, 1e-3, 1e-2])", "reason": "学习率对模型收敛至关重要", "line": 15}}]}}

重要：返回的 JSON 必须在一行内，不要格式化换行。"""
        
        try:
            content = self._call_llm(
                prompt,
                "你是一个专业的神经网络架构搜索（NAS）专家，擅长识别深度学习代码中的可寻优参数。"
            )
            
            # 多层容错 JSON 解析
            return self._extract_candidates(content)
        
        except Exception as e:
            print(f"[LLMClient] Error analyzing code: {e}")
            return []
    
    def _extract_candidates(self, content: str) -> List[Dict]:
        """多层容错提取候选列表"""
        # 尝试1: 直接解析
        try:
            result = json.loads(content)
            if isinstance(result, dict) and "candidates" in result:
                return result["candidates"]
            if isinstance(result, list):
                return result
        except:
            pass
        
        # 尝试2: 从代码块提取对象
        try:
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
            if match:
                result = json.loads(match.group(1))
                if isinstance(result, dict) and "candidates" in result:
                    return result["candidates"]
        except:
            pass
        
        # 尝试3: 直接找 candidates 数组（包含换行）
        try:
            match = re.search(r'"candidates"\s*:\s*(\[[\s\S]*?\])\s*[,}]', content)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        # 尝试4: 找任意 JSON 对象数组
        try:
            match = re.search(r'(\[[\s\S]*"name"[\s\S]*\])', content)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        print(f"[LLMClient] Warning: JSON parse failed")
        print(f"[LLMClient] Content preview: {content[:500]}")
        return []
    
    def resolve_dynamic_reference(self, code: str, variable_name: str) -> str:
        """
        使用真实 LLM 解析动态引用（如 getattr）
        """
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
        """
        使用真实 LLM 生成搜索空间
        """
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
            result = json.loads(content)
            return result.get("search_space", [current_value])
        except:
            return [current_value]
    
    def recommend_injection(self, candidates: List[Dict]) -> List[Dict]:
        """
        使用真实 LLM 推荐哪些参数值得注入
        """
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
            result = json.loads(content)
            return result.get("recommendations", [])
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
