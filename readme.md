# [MVL-HA] Demo 

This repository provides an official implementation of the [MVL-HA] proposed in paper:  
**"Credit Risk Prediction for SMEs based on Multi-view Learning with
Hierarchical Attention Mechanism"**.

## ✨ Key Features
- Core implementation of the MVL-HA algorithm
- Parameter tuning interface with comprehensive logging
- Example dataset for Demo program
- Automated model saving and result tracking

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependency Installation
```bash
# Install required packages
pip install -r requirements.txt
```

## 🚀 Quick Start Guide

### 1. Run the model 

Please see the main.py

### 2. Custom Parameter Tuning
## ⚙️ Tunable Parameters

The following parameters can be customized in `main.py` for different experimental configurations:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `epochs` | `[50]` | Total training iterations |
| `finance_units` | `[30, 60]` | Number of units in financial layer |
| `stock_units` | `[30, 60]` | Number of units in stock layer |
| `network_units` | `[30, 60]` | Number of units in network layer |
| `attention_units` | `[30]` | Dimension of attention mechanism |
| `mlp_units` | `[20]` | Hidden units in MLP classifier |
| `finance_drops` | `[0.3, 0.5]` | Dropout rate for financial features |
| `stock_drops` | `[0.3, 0.5]` | Dropout rate for stock features |
| `network_drops` | `[0.3, 0.5]` | Dropout rate for network features |

### 3. Check Outputs
- **Best Model**: `saved_models`  
- **Detailed Results**:  
  - Best result: `result/best_results_f.csv`  
  - Best parameter: `result/best_parameters_f.txt`  
  - Training logs: `result/multiview.csv`

## 📁 Repository Structure
```
.
├── algorithm.py        # Core algorithm implementation
├── main.py             # Main execution & parameter configuration
├── data/               # Example dataset
├── saved_models/       # Best-performing models (auto-generated)
├── results/            # Detailed experiment results (auto-generated)
├── requirements.txt    # Dependency list
├── README.md           # This documentation
└── utils.py            # Utility functions 
```

## 📜 Citation

If you use this code or reference our work, please cite:

```bibtex
@article{,
  title   = {},
  author  = {},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {},
  doi     = {}
}
```

## 🤝 Contributing & Support

---

*For theoretical details and experimental protocols, please refer to the original paper.*
```
