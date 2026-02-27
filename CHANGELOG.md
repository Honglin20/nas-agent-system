# Changelog

所有显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.3.0] - 2026-02-27

### 新增

#### 配置管理系统
- 新增 `mas_core/config.py` - 完整的配置管理模块
- 支持用户配置 (`~/.nas-cli/config.yaml`)
- 支持项目配置 (`.nas-cli.yaml`)
- 支持环境变量覆盖配置
- 配置优先级：命令行 > 项目配置 > 用户配置 > 默认配置

#### 错误处理系统
- 新增 `mas_core/exceptions.py` - 完善的异常处理
- 定义了 `ErrorCode` 枚举，分类错误码
- 自定义异常层次结构：`NASCLIError`, `FileError`, `LLMError`, `ValidationError`
- 用户友好的错误消息映射

#### 重试和缓存机制
- 新增 `mas_core/retry_cache.py`
- 实现指数退避重试装饰器 `retry_with_backoff`
- 内存缓存 `Cache` 和文件缓存 `FileCache`
- 熔断器模式 `CircuitBreaker` 防止级联故障
- 缓存装饰器 `cached`

#### 备份和撤销系统
- 新增 `mas_core/backup.py`
- 完整的备份管理器 `BackupManager`
- 操作历史记录
- 支持撤销修改 (`nas-cli --undo`)
- 支持恢复到指定备份
- 自动清理旧备份（保留最近10个）

#### LLM 客户端增强
- 支持超时控制
- 支持重试机制（指数退避）
- 支持熔断器模式
- 添加 Mock LLM 客户端用于测试
- 更好的错误分类处理

#### CLI 增强
- 新增 `--undo` 参数撤销修改
- 新增 `--config` 参数编辑配置
- 新增 `--mock` 参数使用 Mock LLM
- 新增 `--verbose` 参数详细输出
- 改进的进度展示（带进度条和耗时）

### 修复

- 移除硬编码的 API Key
- 修复版本号不一致问题（`setup.py` 和代码中统一为 1.3.0）
- 修复文件扫描时的权限检查
- 修复大文件处理（超过 1MB 的文件会被跳过）

### 改进

- 更详细的错误信息和日志
- 更好的输入验证
- 改进的用户提示和反馈
- 支持从环境变量加载更多配置项

## [1.2.0] - 2026-02-26

### 新增

- 智能模型识别（动态反射解析）
- 跨文件参数修改
- LLM 驱动的 Report 插入
- 寻优空间张开

## [1.0.0] - 2026-02-25

### 新增

- 全新交互式 CLI 界面
- 支持 pip 安装
- 智能参数推荐
- 修改差异预览
- 自动备份功能
- 支持 ValueSpace 和 LayerSpace 注入

---

## 版本号说明

- **主版本号**：不兼容的 API 变更
- **次版本号**：向下兼容的功能新增
- **修订号**：向下兼容的问题修复
