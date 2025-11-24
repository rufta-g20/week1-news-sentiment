# 📓 Notebooks

This folder contains all exploratory and experimental Jupyter Notebooks for Week 1 of the News Sentiment Analysis project.

### 📘 Purpose
Notebooks are used for:
- Interactive experimentation
- Exploratory data analysis (EDA)
- **Calling and demonstrating the class methods** from the `scripts/` package.
- Visualizing intermediate results

### 📁 Contents
- `task2_text_eda.ipynb` — Exploration of news headlines, text cleaning, and LDA topic modeling.
- `task2_finance.ipynb` — Financial data loading, calculating returns, and adding TA-Lib indicators.

### ⚠️ Guidelines
- Keep notebooks clean and well-commented.
- Avoid committing large outputs; clear cell outputs before committing.
- **Notebooks must NOT contain core logic** — all reusable functions and business logic must be imported from the **`scripts`** package (e.g., `StockAnalyzer`).

### 🔄 Reproducibility
Before running a notebook, ensure your virtual environment is active:

```bash
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```
Then start Jupyter:
```bash
jupyter notebook
```