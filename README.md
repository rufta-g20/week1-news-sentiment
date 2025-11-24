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
- **Pandas, NumPy, TextBlob, NLTK, Gensim**
- **TA-Lib** (For technical analysis indicators)
- **Object-Oriented Design** (Core logic is encapsulated in **classes** within the `scripts/` package.)

---

## 📁 Repository Structure
week1-news-sentiment/
│
├── .github/workflows/ # CI workflows
│ └── unittests.yml
│
├── .vscode/ # VS Code environment settings
│ └── settings.json
│
├── src/ # Main Python package (core logic lives here)
│ └── init.py
│
├── scripts/ # Core Python Package (Contains StockAnalyzer and NewsCorpusProcessor classes)
│ └── README.md
│
├── notebooks/ # Jupyter notebooks for EDA and experimentation
│ └── README.md
│
├── tests/ # Unit tests to validate project functionality
│ └── init.py
│
├── data/ # Raw and processed datasets (ignored in Git)
│
├── README.md # → You are reading this
├── requirements.txt # Reproducible Python dependencies
├── .gitignore # Ignore unnecessary files / folders
└── venv/ # Virtual environment (ignored)
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
.\venv\Scripts\activate
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
 * Sets up Python
 * Installs project dependencies
 * Runs placeholder tests
Every push and PR automatically triggers CI.

---

# 👩‍💻 Author
**Rufta Gaiem Weldegiorgis** 
**10 Academy — AI Mastery Cohort 8**