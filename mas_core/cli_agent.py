"""
MAS Core - CLI UI Agent
交互智能体：唯一与用户进行终端对话的 Agent
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent


class CLIUIAgent(BaseAgent):
    """
    CLI 交互智能体
    - 展示寻优参数列表
    - 接收用户确认
    - 协调其他 Agent 的交互
    """
    
    def __init__(self):
        super().__init__(
            scope_type="ui",
            scope_name="cli_interface"
        )
        self.pending_candidates: List[Dict[str, Any]] = []
        self.user_choices: Dict[str, Any] = {}
        
    def analyze(self) -> Dict[str, Any]:
        """UI Agent 不需要分析代码"""
        return {'status': 'ui_ready'}
    
    def get_nas_candidates(self) -> List[Dict[str, Any]]:
        """UI Agent 不直接提供候选"""
        return []
    
    def display_candidates(self, candidates: List[Dict[str, Any]], source: str = ""):
        """
        展示 NAS 候选参数列表
        
        Args:
            candidates: 候选参数列表
            source: 来源标识
        """
        self._think(f"Displaying {len(candidates)} NAS candidates from {source}")
        
        print("\n" + "="*70)
        print(f"🔍 NAS Search Space Candidates {'from ' + source if source else ''}")
        print("="*70)
        
        if not candidates:
            print("No candidates found.")
            return
        
        for i, cand in enumerate(candidates, 1):
            print(f"\n[{i}] {cand['name']}")
            print(f"    Type: {cand['type']}")
            print(f"    Current Value: {cand['current_value']}")
            print(f"    Location: Line {cand.get('line', 'unknown')}")
            if 'context' in cand:
                print(f"    Context: {cand['context']}")
            print(f"    Suggested Search Space: {cand['suggestion']}")
        
        print("\n" + "="*70)
        self.pending_candidates = candidates
    
    def get_user_confirmation(self, prompt: str = "Confirm injection?") -> bool:
        """
        获取用户确认
        
        Args:
            prompt: 提示文本
            
        Returns:
            bool: 用户是否确认
        """
        print(f"\n{prompt} (y/n): ", end="")
        # 在实际 CLI 中会等待用户输入
        # 这里模拟确认
        response = "y"  # 默认确认，实际使用时从 stdin 读取
        confirmed = response.lower() in ['y', 'yes']
        
        self._think(f"User confirmation: {'YES' if confirmed else 'NO'}")
        return confirmed
    
    def select_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        让用户选择要注入的候选参数
        
        Args:
            candidates: 所有候选参数
            
        Returns:
            List[Dict[str, Any]]: 用户选择的候选
        """
        self.display_candidates(candidates)
        
        print("\nEnter candidate numbers to inject (comma-separated, or 'all'): ")
        # 模拟用户选择全部
        selection = "all"
        
        if selection.lower() == 'all':
            selected = candidates
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            except:
                selected = candidates
        
        self._think(f"User selected {len(selected)} candidates for injection")
        return selected
    
    def show_modification_plan(self, modifications: List[Dict[str, Any]]):
        """
        展示修改计划
        
        Args:
            modifications: 修改计划列表
        """
        print("\n" + "="*70)
        print("📝 Proposed Code Modifications")
        print("="*70)
        
        for i, mod in enumerate(modifications, 1):
            print(f"\n[{i}] File: {mod.get('file', 'unknown')}")
            print(f"    Line: {mod.get('line', 'unknown')}")
            print(f"    Type: {mod.get('type', 'unknown')}")
            print(f"    Original: {mod.get('original', 'N/A')}")
            print(f"    Modified: {mod.get('modified', 'N/A')}")
        
        print("\n" + "="*70)
        self._think(f"Displayed {len(modifications)} proposed modifications")
    
    def report_success(self, message: str):
        """报告成功"""
        print(f"\n✅ {message}")
        self._think(f"Success: {message}")
    
    def report_error(self, error: str):
        """报告错误"""
        print(f"\n❌ Error: {error}")
        self._think(f"Error: {error}")
    
    def show_progress(self, current: int, total: int, message: str = ""):
        """显示进度"""
        percent = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {percent:.1f}% {message}", end='', flush=True)
        if current == total:
            print()
