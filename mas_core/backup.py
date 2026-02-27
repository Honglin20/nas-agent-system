"""
NAS CLI Backup and Undo System (v1.4.0 Fixed)
备份和撤销系统 - 修复版
- 支持备份描述
- 备份列表展示
- 快速切换备份
- 修复 undo 逻辑
- 添加详细日志
"""
import os
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib


@dataclass
class FileChange:
    """文件变更记录"""
    file_path: str
    original_hash: str
    modified_hash: str
    backup_path: Optional[str] = None
    change_type: str = "modify"  # modify, create, delete


@dataclass
class Operation:
    """操作记录"""
    id: str
    timestamp: float
    description: str
    target_dir: str
    changes: List[FileChange] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'description': self.description,
            'target_dir': self.target_dir,
            'changes': [asdict(c) for c in self.changes],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        changes = [FileChange(**c) for c in data.get('changes', [])]
        return cls(
            id=data['id'],
            timestamp=data['timestamp'],
            description=data['description'],
            target_dir=data['target_dir'],
            changes=changes,
            metadata=data.get('metadata', {})
        )
    
    @property
    def formatted_time(self) -> str:
        """格式化时间"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    @property
    def short_id(self) -> str:
        """短 ID"""
        return self.id[:8]


class BackupManager:
    """备份管理器 - v1.4.0 修复版"""
    
    BACKUP_DIR_NAME = ".nas_backup"
    HISTORY_FILE = "history.json"
    MAX_BACKUPS = 20
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.backup_dir = self.project_path / self.BACKUP_DIR_NAME
        self.history_file = self.backup_dir / self.HISTORY_FILE
        self._operations: List[Operation] = []
        self._load_history()
        
        print(f"[BackupManager v1.4.0] Initialized for project: {project_path}")
        print(f"[BackupManager] Backup directory: {self.backup_dir}")
    
    def _load_history(self):
        """加载操作历史"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._operations = [Operation.from_dict(op) for op in data.get('operations', [])]
                print(f"[BackupManager] Loaded {len(self._operations)} operations from history")
            except Exception as e:
                print(f"[BackupManager] Warning: Failed to load history: {e}")
                self._operations = []
        else:
            print(f"[BackupManager] No history file found, starting fresh")
            self._operations = []
    
    def _save_history(self):
        """保存操作历史"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            data = {
                'operations': [op.to_dict() for op in self._operations],
                'last_updated': time.time()
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[BackupManager] Saved history with {len(self._operations)} operations")
        except Exception as e:
            print(f"[BackupManager] Warning: Failed to save history: {e}")
    
    def _compute_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def create_backup(self, description: str = "", metadata: Optional[Dict] = None) -> Operation:
        """
        创建项目备份 - v1.4.0 修复版
        
        Args:
            description: 备份描述（必填，用于识别备份内容）
            metadata: 额外元数据
            
        Returns:
            Operation: 操作记录
        """
        operation_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
        timestamp = time.time()
        
        # v1.4.0: 如果没有描述，自动生成一个
        if not description:
            description = f"Backup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print(f"[BackupManager] Creating backup: {operation_id}")
        print(f"[BackupManager] Description: {description}")
        
        operation = Operation(
            id=operation_id,
            timestamp=timestamp,
            description=description,
            target_dir=str(self.project_path),
            metadata=metadata or {}
        )
        
        # 创建备份目录
        backup_subdir = self.backup_dir / operation_id
        backup_subdir.mkdir(parents=True, exist_ok=True)
        print(f"[BackupManager] Backup directory: {backup_subdir}")
        
        # 备份所有 Python 文件
        backed_up_count = 0
        for py_file in self.project_path.rglob("*.py"):
            # 跳过备份目录自身
            if self.BACKUP_DIR_NAME in py_file.parts:
                continue
            
            # 跳过隐藏目录
            if any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                rel_path = py_file.relative_to(self.project_path)
                backup_path = backup_subdir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制文件
                shutil.copy2(py_file, backup_path)
                
                # 记录变更
                original_hash = self._compute_hash(py_file)
                change = FileChange(
                    file_path=str(rel_path),
                    original_hash=original_hash,
                    modified_hash=original_hash,  # 备份时相同
                    backup_path=str(backup_path),
                    change_type="backup"
                )
                operation.changes.append(change)
                backed_up_count += 1
                
            except Exception as e:
                print(f"[BackupManager] Warning: Failed to backup {py_file}: {e}")
        
        # 备份 YAML 配置文件
        for yaml_file in self.project_path.rglob("*.yaml"):
            if self.BACKUP_DIR_NAME in yaml_file.parts:
                continue
            if any(part.startswith('.') for part in yaml_file.parts):
                continue
            
            try:
                rel_path = yaml_file.relative_to(self.project_path)
                backup_path = backup_subdir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(yaml_file, backup_path)
                
                original_hash = self._compute_hash(yaml_file)
                change = FileChange(
                    file_path=str(rel_path),
                    original_hash=original_hash,
                    modified_hash=original_hash,
                    backup_path=str(backup_path),
                    change_type="backup"
                )
                operation.changes.append(change)
                backed_up_count += 1
            except Exception as e:
                print(f"[BackupManager] Warning: Failed to backup {yaml_file}: {e}")
        
        # 保存操作记录
        self._operations.append(operation)
        self._cleanup_old_backups()
        self._save_history()
        
        print(f"[BackupManager] Created backup: {operation_id}")
        print(f"[BackupManager] Backed up {backed_up_count} files")
        
        return operation
    
    def record_modification(self, 
                           file_path: Path, 
                           original_hash: str,
                           modified_hash: str,
                           operation_id: Optional[str] = None) -> bool:
        """
        记录文件修改
        
        Args:
            file_path: 文件路径
            original_hash: 原始哈希
            modified_hash: 修改后哈希
            operation_id: 关联的操作ID
            
        Returns:
            bool: 是否成功
        """
        try:
            rel_path = file_path.relative_to(self.project_path)
            
            change = FileChange(
                file_path=str(rel_path),
                original_hash=original_hash,
                modified_hash=modified_hash,
                change_type="modify"
            )
            
            # 查找或创建操作
            if operation_id:
                operation = next((op for op in self._operations if op.id == operation_id), None)
                if operation:
                    operation.changes.append(change)
                    self._save_history()
                    print(f"[BackupManager] Recorded modification for {rel_path} in operation {operation_id}")
                    return True
            
            # 创建新操作
            operation = Operation(
                id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12],
                timestamp=time.time(),
                description=f"Modification at {datetime.now().isoformat()}",
                target_dir=str(self.project_path),
                changes=[change]
            )
            self._operations.append(operation)
            self._save_history()
            print(f"[BackupManager] Created new operation for modification: {operation.id}")
            
            return True
            
        except Exception as e:
            print(f"[BackupManager] Warning: Failed to record modification: {e}")
            return False
    
    def undo(self, operation_id: Optional[str] = None) -> bool:
        """
        撤销操作 - v1.4.0 修复版
        
        Args:
            operation_id: 要撤销的操作ID，None 表示撤销最后一次操作
            
        Returns:
            bool: 是否成功
        """
        print(f"[BackupManager] Starting undo operation...")
        
        if not self._operations:
            print("[BackupManager] No operations to undo")
            return False
        
        # 查找要撤销的操作
        if operation_id:
            operation = next((op for op in self._operations if op.id == operation_id), None)
            if not operation:
                print(f"[BackupManager] Operation not found: {operation_id}")
                return False
        else:
            # 找到最后一个未撤销的操作
            operation = None
            for op in reversed(self._operations):
                if not op.metadata.get('undone', False):
                    operation = op
                    break
            
            if not operation:
                print("[BackupManager] No operations to undo (all already undone)")
                return False
        
        print(f"[BackupManager] Undoing operation: {operation.id}")
        print(f"[BackupManager] Description: {operation.description}")
        print(f"[BackupManager] Target directory: {operation.target_dir}")
        
        # 验证备份目录存在
        backup_subdir = self.backup_dir / operation.id
        if not backup_subdir.exists():
            print(f"[BackupManager] ERROR: Backup directory not found: {backup_subdir}")
            return False
        
        print(f"[BackupManager] Backup directory: {backup_subdir}")
        
        success_count = 0
        fail_count = 0
        
        for change in operation.changes:
            file_path = self.project_path / change.file_path
            backup_file = backup_subdir / change.file_path
            
            print(f"[BackupManager] Processing: {change.file_path}")
            print(f"[BackupManager]   Target: {file_path}")
            print(f"[BackupManager]   Backup: {backup_file}")
            
            if backup_file.exists():
                try:
                    # 确保目标目录存在
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 恢复文件
                    shutil.copy2(backup_file, file_path)
                    print(f"[BackupManager]   ✓ Restored: {change.file_path}")
                    success_count += 1
                except Exception as e:
                    print(f"[BackupManager]   ✗ Failed to restore {change.file_path}: {e}")
                    fail_count += 1
            else:
                print(f"[BackupManager]   ✗ Backup not found: {backup_file}")
                fail_count += 1
        
        # 标记操作为已撤销
        operation.metadata['undone'] = True
        operation.metadata['undone_at'] = time.time()
        self._save_history()
        
        print(f"[BackupManager] Undo complete: {success_count} succeeded, {fail_count} failed")
        return fail_count == 0
    
    def list_operations(self) -> List[Operation]:
        """列出所有操作"""
        return self._operations.copy()
    
    def get_operation(self, operation_id: str) -> Optional[Operation]:
        """获取指定操作"""
        return next((op for op in self._operations if op.id == operation_id), None)
    
    def list_backups_with_info(self) -> List[Dict[str, Any]]:
        """
        v1.4.0: 列出所有备份及其详细信息
        
        Returns:
            List[Dict]: 备份信息列表
        """
        backups = []
        for op in self._operations:
            info = {
                'id': op.id,
                'short_id': op.short_id,
                'timestamp': op.timestamp,
                'formatted_time': op.formatted_time,
                'description': op.description,
                'file_count': len(op.changes),
                'target_dir': op.target_dir,
                'metadata': op.metadata,
                'undone': op.metadata.get('undone', False)
            }
            backups.append(info)
        return backups
    
    def display_backup_list(self):
        """
        v1.4.0: 显示备份列表（用于 CLI 展示）
        """
        backups = self.list_backups_with_info()
        
        if not backups:
            print("\n[BackupManager] 没有找到任何备份")
            return
        
        print("\n" + "=" * 80)
        print(f"{'ID':<10} {'时间':<20} {'文件数':<8} {'描述':<30} {'状态'}")
        print("=" * 80)
        
        for i, backup in enumerate(backups, 1):
            status = "已撤销" if backup['undone'] else "可用"
            desc = backup['description'][:28] + ".." if len(backup['description']) > 30 else backup['description']
            print(f"{backup['short_id']:<10} {backup['formatted_time']:<20} "
                  f"{backup['file_count']:<8} {desc:<30} {status}")
        
        print("=" * 80)
        print(f"共 {len(backups)} 个备份")
    
    def switch_to_backup(self, operation_id: str) -> bool:
        """
        v1.4.0: 切换到指定备份（先备份当前状态，再恢复指定版本）
        
        Args:
            operation_id: 目标备份 ID
            
        Returns:
            bool: 是否成功
        """
        # 先备份当前状态
        current_backup = self.create_backup(
            description=f"Auto-backup before switching to {operation_id[:8]}"
        )
        print(f"[BackupManager] 已自动备份当前状态: {current_backup.short_id}")
        
        # 然后恢复到指定备份
        return self.restore_backup(operation_id)
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        if len(self._operations) <= self.MAX_BACKUPS:
            return
        
        # 保留最近的备份
        to_remove = self._operations[:-self.MAX_BACKUPS]
        self._operations = self._operations[-self.MAX_BACKUPS:]
        
        for op in to_remove:
            backup_subdir = self.backup_dir / op.id
            if backup_subdir.exists():
                try:
                    shutil.rmtree(backup_subdir)
                    print(f"[BackupManager] Cleaned up old backup: {op.id}")
                except Exception as e:
                    print(f"[BackupManager] Failed to cleanup {op.id}: {e}")
    
    def restore_backup(self, operation_id: str) -> bool:
        """
        恢复到指定备份 - v1.4.0 修复版
        
        Args:
            operation_id: 备份操作ID
            
        Returns:
            bool: 是否成功
        """
        print(f"[BackupManager] Restoring to backup: {operation_id}")
        
        operation = self.get_operation(operation_id)
        if not operation:
            print(f"[BackupManager] Operation not found: {operation_id}")
            return False
        
        backup_subdir = self.backup_dir / operation_id
        if not backup_subdir.exists():
            print(f"[BackupManager] Backup directory not found: {operation_id}")
            return False
        
        print(f"[BackupManager] Description: {operation.description}")
        
        success_count = 0
        fail_count = 0
        
        for change in operation.changes:
            backup_file = backup_subdir / change.file_path
            target_file = self.project_path / change.file_path
            
            if backup_file.exists():
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
                    success_count += 1
                    print(f"[BackupManager] Restored: {change.file_path}")
                except Exception as e:
                    print(f"[BackupManager] Failed to restore {change.file_path}: {e}")
                    fail_count += 1
            else:
                print(f"[BackupManager] Backup file not found: {backup_file}")
                fail_count += 1
        
        print(f"[BackupManager] Restore complete: {success_count} succeeded, {fail_count} failed")
        return fail_count == 0
    
    def verify_backup(self, operation_id: str) -> bool:
        """
        验证备份完整性
        
        Args:
            operation_id: 备份操作ID
            
        Returns:
            bool: 是否完整
        """
        operation = self.get_operation(operation_id)
        if not operation:
            print(f"[BackupManager] Operation not found: {operation_id}")
            return False
        
        backup_subdir = self.backup_dir / operation_id
        if not backup_subdir.exists():
            print(f"[BackupManager] Backup directory not found: {operation_id}")
            return False
        
        all_valid = True
        for change in operation.changes:
            backup_file = backup_subdir / change.file_path
            if not backup_file.exists():
                print(f"[BackupManager] Missing file: {change.file_path}")
                all_valid = False
                continue
            
            # 验证哈希
            current_hash = self._compute_hash(backup_file)
            if current_hash != change.original_hash:
                print(f"[BackupManager] Hash mismatch: {change.file_path}")
                all_valid = False
        
        if all_valid:
            print(f"[BackupManager] Backup {operation_id} is valid")
        
        return all_valid
    
    def delete_backup(self, operation_id: str) -> bool:
        """
        v1.4.0: 删除指定备份
        
        Args:
            operation_id: 备份操作ID
            
        Returns:
            bool: 是否成功
        """
        operation = self.get_operation(operation_id)
        if not operation:
            print(f"[BackupManager] Operation not found: {operation_id}")
            return False
        
        # 删除备份目录
        backup_subdir = self.backup_dir / operation_id
        if backup_subdir.exists():
            try:
                shutil.rmtree(backup_subdir)
                print(f"[BackupManager] Deleted backup directory: {operation_id}")
            except Exception as e:
                print(f"[BackupManager] Failed to delete backup directory: {e}")
                return False
        
        # 从历史记录中移除
        self._operations = [op for op in self._operations if op.id != operation_id]
        self._save_history()
        
        print(f"[BackupManager] Deleted backup: {operation_id}")
        return True
    
    def get_latest_backup_id(self) -> Optional[str]:
        """
        v1.4.0: 获取最新的未撤销备份 ID
        
        Returns:
            备份 ID 或 None
        """
        for op in reversed(self._operations):
            if not op.metadata.get('undone', False):
                return op.id
        return None
