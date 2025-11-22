# ⚙️ Scripts Folder

This directory contains small utility scripts and helpers used throughout the project.

### 📘 Purpose
The scripts here support tasks such as:
- Data loading or preprocessing
- Sentiment analysis utilities
- Model preparation helpers
- Reusable functions for notebooks

### 📁 Contents
- README.md — Documentation
- *(Add scripts as project progresses)*

### 🧩 Best Practices
- Keep scripts modular and reusable.
- Avoid placing notebook-specific logic here.
- Add docstrings to every function.

### 🧪 Testing
Any script with important logic should have a corresponding test inside tests/.

Example:
```bash
scripts/ sentiment_utils.py tests/ test_sentiment_utils.py
```