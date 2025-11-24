# 📰 Week 1 — News Sentiment Analysis Pipeline  
10 Academy — Artificial Intelligence Mastery Program  

This project implements a **news sentiment analysis system** using Python, Git, and CI automation. It is based on Week 1 of the 10 Academy AI Mastery Program and focuses on building clean, reproducible, and scalable project structure.

---

## 🌟 Project Overview
The Week 1 challenge aims to teach strong software engineering foundations through:

- Proper **Git branching workflows**
- Creating a clean **Python environment**
- Organizing a production-ready **folder structure**
- Implementing **continuous integration (CI)** with GitHub Actions
- Preparing a solid base for sentiment analysis and time-series prediction in the next tasks

This repository contains all required setup, with further implementation to be added in Task 2 & 3.

---

## 🛠️ Tech Stack
- **Python 3.11+**
- **VS Code**
- **Git & GitHub**
- **GitHub Actions (CI)**
- **Pandas, NumPy, TextBlob, NLTK, Gensim, yfinance**
- **TA-Lib** (For technical analysis indicators)
- **Object-Oriented Design** (Core logic is encapsulated in **classes** within the `scripts/` package.)

---

## 📁 Repository Structure

week1-news-sentiment/ 
 │ ├── .github/workflows/ # CI workflows
 │   └── unittests.yml
 |
 │ ├── .vscode/ # VS Code environment settings 
 │   └── settings.json 
 |
 │ ├── src/ # Main Python package (core logic lives here) 
 │   └── init.py 
 |
 │ ├── scripts/ # Utility scripts for automation or data processing 
 │   ├── README.md 
 |   ├── init.py
 │   ├── finance_tools.py # StockAnalyzer Class 
 │   └── text_processing.py # NewsCorpusProcessor Class 
 |
 │ ├── notebooks/ # Jupyter notebooks for EDA and experimentation 
 │   ├── README.md 
 │   ├── task2_finance.ipynb 
 │   └── task2_text_eda.ipynb 
 |
 │ ├── tests/ # Unit tests to validate project functionality 
 │   ├── init.py 
 |   ├── README.md
 │   ├── test_smoke_task2.py # End-to-end smoke test
 |   ├── test_text_processing.py # Unit tests for text processing
 │   └── test_finance_tools.py # Unit tests for financial tools 
 |
 │ ├── data/ # Raw and processed datasets (ignored in Git) 
 │ ├── README.md # → You are reading this 
 | ├── requirements.txt # Reproducible Python dependencies 
 | ├── .gitignore # Ignore unnecessary files / folders 
 | └── venv/ # Virtual environment (ignored)
---

## 🚀 Key Modules Usage Example

​The project's core functionality is encapsulated in the `StockAnalyzer` and `NewsCorpusProcessor` classes, located in the `scripts/` folder.

### 📈 Using the `StockAnalyzer`
​This example demonstrates loading data and adding technical indicators:

```bash
from scripts.finance_tools import StockAnalyzer
import pandas as pd

# 1. Initialize the Analyzer (loads data automatically)
start_date = '2023-01-01'
end_date = '2024-01-01'
analyzer = StockAnalyzer(ticker='MSFT', start=start_date, end=end_date)

print(f"Loaded {analyzer.ticker} data from {start_date} to {end_date}.")
print(f"Initial rows: {analyzer.df.shape[0]}")

# 2. Add Technical Indicators
analyzer.add_indicators()

# 3. View the results (last 5 rows with indicators)
print("\nDataFrame with Indicators (Last 5 rows):")
print(analyzer.df[['Close', 'SMA_20', 'RSI_14', 'MACD', 'MACDSignal']].tail())

# The result is stored in analyzer.df
```
---

## 🔧 Environment Setup (Reproducible Steps)

Follow these steps to reproduce my environment on any Windows or Linux machine:

### 1️⃣ Clone the repository
```bash
git clone [https://github.com/rufta-g20/week1-news-sentiment.git](https://github.com/rufta-g20/week1-news-sentiment.git)
cd week1-news-sentiment
```

### 2️⃣ Create & activate the virtual environment
Windows PowerShell
```bash
python -m venv venv
.\\venv\\Scripts\\activate
```
Linux/Mac
```bash
python3 -m venv venv 
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Optional: Start Jupyter
```bash
jupyter notebook
```

---

## ⚙️ Continuous Integration (CI)
This repository includes a GitHub Actions workflow *(unittests.yml)* that:
 * Sets up Python and installs TA-Lib dependencies (required for ta-lib).
 * Installs all dependencies including pytest.
 * Automatically runs all unit and smoke tests in the `tests/` folder.
 
 Every push and PR automatically triggers CI.

---

# 👩‍💻 Author
**Rufta Gaiem Weldegiorgis** 

**10 Academy — AI Mastery Cohort 8**