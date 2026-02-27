# NAS-CLI v1.0.0

智能神经网络架构搜索（NAS）寻优空间注入 CLI 工具。

## 功能特性

- 🧠 **智能代码分析**：自动识别代码中的可寻优参数
- 💬 **交互式界面**：友好的命令行交互体验
- 📁 **项目导航**：支持 Tab 补全的目录选择
- 🔍 **架构扫描**：从入口文件开始扫描整个项目
- ⭐ **智能推荐**：自动推荐值得寻优的参数
- 📝 **差异预览**：修改前显示清晰的对比
- 💾 **自动备份**：修改前自动备份原文件

## 安装

```bash
# 从源码安装
pip install -e .

# 或使用 requirements
pip install -r requirements.txt
```

## 使用方法

安装完成后，在命令行输入：

```bash
nas-cli
```

### 使用流程

1. **选择项目目录**
   - 支持 Tab 补全
   - 显示目录预览

2. **选择入口脚本**
   - 自动发现 Python 文件
   - 智能推荐入口文件

3. **扫描项目架构**
   - 实时显示扫描进度
   - 展示项目结构和统计

4. **配置寻优空间**
   - 显示所有候选参数
   - 智能推荐（⭐标记）
   - 支持多种选择方式

5. **确认修改**
   - 显示修改前后差异
   - 确认后执行注入

6. **完成**
   - 自动备份原文件
   - 显示修改摘要

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
nas-cli
# 选择 target_projects/level1 目录
# 选择 train.py 作为入口
```

### Level 2: 跨文件静态传参
```bash
nas-cli
# 选择 target_projects/level2 目录
# 选择 main.py 作为入口
```

### Level 3: 动态反射与 YAML
```bash
nas-cli
# 选择 target_projects/level3 目录
# 选择 main.py 作为入口
```

## 版本历史

### v1.0.0 (2026-02-26)
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
```

## License

MIT
