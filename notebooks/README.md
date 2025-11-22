# 📓 Notebooks

This folder contains all exploratory and experimental Jupyter Notebooks for Week 1 of the News Sentiment Analysis project.

### 📘 Purpose
Notebooks are used for:
- Interactive experimentation
- Exploratory data analysis (EDA)
- Prototyping data-cleaning or modeling logic
- Visualizing intermediate results

### 📁 Contents
- week1_experiment.ipynb *(example)* — Prototyping and analysis notebook(s)

### ⚠️ Guidelines
- Keep notebooks clean and well-commented.
- Avoid committing large outputs; clear cell outputs before committing.
- Notebooks should NOT contain production logic — move reusable code to src/ or scripts/.

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