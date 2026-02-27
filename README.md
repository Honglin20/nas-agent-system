# NAS-CLI v1.3.0

智能神经网络架构搜索（NAS）寻优空间注入 CLI 工具。

## 功能特性

- 🧠 **智能代码分析**：自动识别代码中的可寻优参数
- 💬 **交互式界面**：友好的命令行交互体验
- 📁 **项目导航**：支持 Tab 补全的目录选择
- 🔍 **架构扫描**：从入口文件开始扫描整个项目
- ⭐ **智能推荐**：自动推荐值得寻优的参数
- 📝 **差异预览**：修改前显示清晰的对比
- 💾 **自动备份**：修改前自动备份原文件
- 🔄 **撤销/重做**：支持撤销修改操作
- ⚙️ **配置持久化**：支持用户配置和项目配置
- 🛡️ **完善的错误处理**：分类错误码，用户友好的错误消息
- 🔄 **重试机制**：LLM 调用失败自动重试
- ⏱️ **超时控制**：防止长时间阻塞

## 安装

```bash
# 从源码安装
pip install -e .

# 或使用 requirements
pip install -r requirements.txt
```

## 快速开始

### 1. 设置 API Key

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

### 2. 运行工具

```bash
nas-cli
```

### 3. 按照交互提示操作

1. 选择项目目录
2. 选择入口脚本
3. 扫描项目架构
4. 选择要寻优的参数
5. 确认修改

## 使用方法

### 基本用法

```bash
# 启动交互式界面
nas-cli

# 指定目录和入口文件
nas-cli --dir ./my_project --entry main.py

# 显示版本
nas-cli --version

# 详细输出模式
nas-cli --verbose

# 使用 Mock LLM (测试模式)
nas-cli --mock
```

### 撤销修改

```bash
# 撤销上次修改
nas-cli --undo --dir ./my_project

# 或使用 nas-agent 命令
nas-agent undo ./my_project
```

### 配置管理

```bash
# 显示当前配置
nas-agent config --show

# 编辑配置文件
nas-agent config --edit
# 或
nas-cli --config

# 重置为默认配置
nas-agent config --reset
```

### 分析项目

```bash
nas-agent analyze ./my_project
```

### 测试靶机项目

```bash
nas-agent test --level 1
nas-agent test --level 2
nas-agent test --level 3
```

## 配置文件

### 用户配置

配置文件位于 `~/.nas-cli/config.yaml`：

```yaml
version: "1.3.0"

llm:
  base_url: "https://api.openai.com/v1"
  models:
    - "moonshot-v1-128k"
    - "moonshot-v1-32k"
    - "moonshot-v1-8k"
  timeout: 60
  max_retries: 3
  retry_delay: 1.0
  temperature: 0.2

ui:
  theme: "default"
  show_progress: true
  confirm_before_modify: true
  auto_backup: true
  verbose: false
  language: "zh"

analysis:
  exclude_patterns:
    - "__pycache__"
    - ".git"
    - "venv"
    - "env"
  include_patterns:
    - "*.py"
  max_file_size: 1048576
  use_cache: true
  cache_ttl: 3600

nas:
  value_keywords:
    - "lr"
    - "learning_rate"
    - "batch_size"
    - "epoch"
    - "dropout"
  layer_keywords:
    - "activation"
    - "optimizer"
    - "norm"
    - "loss"
```

### 项目配置

在项目根目录创建 `.nas-cli.yaml`：

```yaml
# 项目级配置会覆盖用户配置
analysis:
  exclude_patterns:
    - "third_party"
    - "vendor"

nas:
  value_keywords:
    - "custom_param"
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | LLM API Key | - |
| `OPENAI_BASE_URL` | LLM API URL | `https://api.openai.com/v1` |
| `NAS_CLI_LLM_TIMEOUT` | LLM 超时时间(秒) | 60 |
| `NAS_CLI_LLM_MAX_RETRIES` | LLM 最大重试次数 | 3 |
| `NAS_CLI_VERBOSE` | 详细输出模式 | false |
| `NAS_CLI_LANGUAGE` | 界面语言 | zh |

## 支持的参数类型

### ValueSpace（数值寻优）
- 学习率 (learning_rate, lr)
- 批次大小 (batch_size)
- Dropout 率 (dropout_rate)
- 模型维度 (d_model, hidden_dim)
- 层数 (num_layers)
- 训练轮数 (num_epochs)
- 权重衰减 (weight_decay)

### LayerSpace（层选择）
- 激活函数 (ReLU, Sigmoid, Tanh, GELU)
- 优化器 (Adam, SGD)
- 归一化层 (BatchNorm, LayerNorm)

## 项目结构

```
nas-agent-system/
├── nas_cli/              # CLI 主程序
│   ├── __init__.py
│   └── main.py           # 交互式 CLI 实现
├── mas_core/             # MAS 核心架构
│   ├── registry.py       # 中心注册表
│   ├── scope_agent.py    # 作用域智能体
│   ├── modifier_agent.py # 代码修改智能体
│   ├── llm_client.py     # LLM 客户端
│   ├── config.py         # 配置管理 (v1.3.0)
│   ├── exceptions.py     # 异常处理 (v1.3.0)
│   ├── retry_cache.py    # 重试和缓存 (v1.3.0)
│   ├── backup.py         # 备份管理 (v1.3.0)
│   └── ...
├── target_projects/      # 测试靶机项目
│   ├── level1/          # 静态单文件
│   ├── level2/          # 跨文件传参
│   └── level3/          # 动态反射
└── setup.py
```

## 靶机项目测试

### Level 1: 静态单文件
```bash
nas-cli --dir target_projects/level1 --entry train.py
```

### Level 2: 跨文件静态传参
```bash
nas-cli --dir target_projects/level2 --entry main.py
```

### Level 3: 动态反射与 YAML
```bash
nas-cli --dir target_projects/level3 --entry main.py
```

## 版本历史

### v1.3.0 (2026-02-27)
- ✨ 完善的错误处理系统，分类错误码
- ✨ 配置持久化支持（用户配置和项目配置）
- ✨ 撤销/重做功能
- ✨ LLM 调用重试机制和超时控制
- ✨ 熔断器模式防止级联故障
- ✨ Mock LLM 客户端用于测试
- ✨ 详细的进度展示
- ✨ 命令历史记录
- 🐛 修复版本号不一致问题
- 🐛 移除硬编码的 API Key

### v1.2.0 (2026-02-26)
- ✨ 智能模型识别（动态反射解析）
- ✨ 跨文件参数修改
- ✨ LLM 驱动的 Report 插入
- ✨ 寻优空间张开

### v1.0.0 (2026-02-25)
- ✨ 全新交互式 CLI 界面
- ✨ 支持 pip 安装
- ✨ 智能参数推荐
- ✨ 修改差异预览
- ✨ 自动备份功能

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest tests/

# 代码格式化
black mas_core/ nas_cli/

# 类型检查
mypy mas_core/
```

## License

MIT
