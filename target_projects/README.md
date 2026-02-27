# Target Projects for NAS Agent System

This directory contains three levels of target projects to test the MAS (Multi-Agent System) CLI tool.

## Level 1: Static Single File
**File:** `level1/train.py`

All configurations are hard-coded in a single file.
- Model architecture parameters (d_model, hidden_dim, num_layers, dropout_rate)
- Activation function (nn.ReLU())
- Training hyperparameters (batch_size, learning_rate, num_epochs, weight_decay)

**Challenge:** Extract and replace hard-coded values with NAS search spaces.

## Level 2: Cross-File Static Parameters
**Files:** `level2/main.py`, `level2/model.py`, `level2/config.yaml`

Parameters are defined in dictionaries in `main.py` and passed to `model.py`.
- Model configuration dictionary
- Training configuration dictionary
- String-based activation selection
- Conditional optimizer selection

**Challenge:** Trace parameter flow across files, handle dictionary-based configs.

## Level 3: Dynamic Reflection with YAML
**Files:** `level3/main.py`, `level3/models.py`, `level3/config.yaml`

Full dynamic loading using `getattr()` and YAML configuration.
- Model class loaded dynamically via `getattr(models, model_name)`
- All parameters from YAML config
- Dynamic optimizer and loss function selection
- Complex model architectures (ResidualModel, DynamicModel)

**Challenge:** Resolve dynamic imports, handle complex nested configurations, trace through reflection.

## Usage

```bash
# Level 1
cd level1
python train.py

# Level 2
cd level2
python main.py

# Level 3
cd level3
python main.py
```
