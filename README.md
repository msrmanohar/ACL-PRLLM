# Prompt Recovery for LLM in Telugu
## Mano_sub@DravidianLangTech@ACL 2026

Code for the paper:
**"Article-Aware Batching and Discriminative Fine-Tuning of MuRIL for Telugu Prompt-Style Classification"**

📄 Paper: DravidianLangTech@ACL 2026
🏆 Ranked 12th in the shared task

---

## Dataset
The dataset is too large for GitHub. Download from Hugging Face:
👉 https://huggingface.co/datasets/msrmanohar/telugu-prompt-style-recovery

| File | Description |
|------|-------------|
| `train.csv` | 3000 labeled training samples |
| `dev.csv` | Development/validation samples |
| `test_unlabeled.csv` | Test samples (no labels) |

---

## Setup
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 telugu_style_fixed.py
```

## Requirements
- Python 3.12
- PyTorch 2.10
- Transformers 4.40
- scikit-learn

---

## Citation
```
@inproceedings{prompt-telugu-dravidianlangtech-acl-2026,
  title={Shared Task on Prompt Recovery for LLM in Telugu},
  author={B, Premjith and G, Jyothish Lal and Bharathi Raja Chakravarthi and Saranya Rajiakodi and Durairaj, Thenmozhi and Ratnavel Rajalakshmi and Rahul Ponnusamy and Chinthala Bhuvanesh},
  booktitle={Proceedings of the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```
