# Ciphertext-Only Classification of Encryption Algorithms Using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Source code for the paper **"Ciphertext-Only Classification of Encryption Algorithms Using Deep Learning"** submitted to the *Journal of Intelligent Systems*.

## 📄 Abstract

This repository contains the complete implementation of a deep learning approach for identifying encryption algorithms (AES, 3DES, Blowfish, Kuznechik) from ciphertext-only data. The study demonstrates that classification accuracy depends critically on cryptographic parameter reuse: under key reuse, the model achieves 96.09% accuracy, but under proper key rotation, accuracy collapses to 33.33% (random baseline), validating the security of modern block ciphers under correct key management.

## 🚀 Quick Start

### Prerequisites

- **Python 3.12** (required)
- GPU recommended (training takes ~2 hours total on GPU, ~8 hours on CPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/malikzh/cypher-detector.git
cd cypher-detector

# Install dependencies
pip install -r requirements.txt
```

### Reproduction Steps

```bash
# 1. Generate dataset (~5 minutes, creates _dataset/ folder)
./datagen.py

# 2. Train Scenario A: Key-reuse with random split (~1 hour on GPU)
./train_scenario_a.py

# 3. Train Scenario B: Key-disjoint split (~1 hour on GPU)
./train_scenario_b.py

# 4. Generate accuracy plots
./plot_accuracy.py

# 5. Generate confusion matrices
./plot_confusion.py
```

## 📊 Expected Results

| Scenario | Description | Train Acc | Val Acc |
|----------|-------------|-----------|---------|
| **A: Key-reuse** | Same keys in train/val | 97.65% | **96.09%** |
| **B: Key-disjoint** | Zero key overlap | 99.91% | **33.33%** |
| Random baseline | 4 classes | — | 25.00% |

**Key Finding:** The 62.76 percentage point accuracy drop in Scenario B demonstrates that high performance under key reuse reflects memorization of key-specific artifacts, not algorithm-intrinsic features. This validates that modern ciphers remain secure under proper key management.

## 📁 Repository Structure

```
cypher-detector/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── datagen.py                   # Dataset generation script
├── train_scenario_a.py          # Training for Scenario A
├── train_scenario_b.py          # Training for Scenario B
├── plot_accuracy.py             # Generate accuracy/loss plots
├── plot_confusion.py            # Generate confusion matrices
├── model.py                     # CNN-Transformer architecture
├── dataset.py                   # PyTorch Dataset class
├── config.py                    # Hyperparameters and configuration
├── encoder/                     # Cipher implementations
│   ├── __init__.py
│   ├── aes.py                   # AES encoder
│   ├── twofish.py               # Blowfish encoder
│   ├── triple_des.py            # 3DES encoder
│   └── kuznechik.py             # Kuznechik encoder
└── _dataset/                    # Generated dataset (not in repo)
    ├── AES/
    ├── 3DES/
    ├── Blowfish/
    └── Kuznechik/
```

## 🔬 Dataset Details

- **Total samples:** 4,096 (1,024 per cipher)
- **Ciphertext length:** 1,024 bytes
- **Ciphers:** AES-256, 3DES, Blowfish-256, Kuznechik-256 (all in CBC mode)
- **Keys:** 32 unique keys per cipher
- **Key reuse factor:** 32:1 (each key encrypts 32 plaintexts)
- **IV policy:** Single fixed IV per cipher class
- **Padding:** None (plaintexts are exact multiples of block size)
- **Libraries:** `cryptography` (AES, 3DES, Blowfish), `gostcrypto` (Kuznechik)

## 🧠 Model Architecture

**CNN-Transformer Hybrid:**
- Embedding layer (vocab_size=256, d_model=16)
- 1D Convolutional layer (in=16, out=32, kernel=3)
- Transformer Encoder (d_model=32, num_layers=2, nhead=4)
- Fully connected classifier (hidden=32, output=4 classes)

**Training:**
- Loss: Categorical cross-entropy
- Optimizer: AdamW
- Regularization: Dropout (p=0.2)
- Epochs: 50
- Batch size: 64

## 📈 Visualizations

After running the scripts, the following files will be generated:

- `val_accuracy_scenario_a.pdf` — Training/validation accuracy curves (Scenario A)
- `val_accuracy_scenario_b.pdf` — Training/validation accuracy curves (Scenario B)
- `val_loss_scenario_a.pdf` — Training/validation loss curves (Scenario A)
- `val_loss_scenario_b.pdf` — Training/validation loss curves (Scenario B)
- `confusion_scenario_a.pdf` — 4×4 confusion matrix (Scenario A)
- `confusion_scenario_b.pdf` — 4×4 confusion matrix (Scenario B)

## ⚙️ Hardware Requirements

- **Minimum:** CPU-only (training takes ~8 hours total)
- **Recommended:** NVIDIA GPU with CUDA support (training takes ~2 hours total)
- **Memory:** 8GB RAM minimum, 16GB recommended
- **Storage:** ~1GB for dataset and models

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'gostcrypto'`  
**Solution:** Install gostcrypto: `pip install gostcrypto`

**Issue:** Dataset generation fails  
**Solution:** Ensure you have write permissions in the current directory

**Issue:** Training is very slow on CPU  
**Solution:** This is expected. Consider using a GPU or reducing the number of epochs in `config.py`

**Issue:** CUDA out of memory  
**Solution:** Reduce batch size in `config.py` (e.g., from 64 to 32)

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{alimzhanova2025ciphertext,
  title={Ciphertext-Only Classification of Encryption Algorithms Using Deep Learning},
  author={Alimzhanova, Zhanna and Zharykov, Malik and Zhunusbayeva, Salamat and Tulesheva, Gulnara and Smatova, Gulzhazira},
  journal={Journal of Intelligent Systems},
  year={2025},
  note={Under review}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Cryptographic implementations based on [PyCryptodome](https://www.pycryptodome.org/) and [gostcrypto](https://github.com/drobotun/gostcrypto)
- Neural network architecture inspired by [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:

- **Malik Zharykov** - zharykov.m@stud.satbayev.university

---

**Note:** This research demonstrates that proper key management (frequent key rotation) is essential for maintaining cipher security. The results validate cryptographic best practices rather than reveal vulnerabilities in modern block ciphers.
