# ⚙️ Scripts Folder

This directory contains **core business logic** for the project, encapsulated in reusable Python classes.

### 📘 Purpose
The scripts here support tasks such as:
- **Object-Oriented Design:** Encapsulating related functionality into classes.
- Data loading, cleaning, and transformation logic.
- Reusable components for analysis notebooks.

### 📁 Contents & Core Components
- `__init__.py` — Makes `scripts/` a Python package, enabling direct imports.
- `finance_tools.py` — Contains the **`StockAnalyzer` class** for loading stock data and computing technical indicators.
- `text_processing.py` — Contains the **`NewsCorpusProcessor` class** for text cleaning, corpus preparation, and LDA topic modeling.
- `README.md` — Documentation (You are here).

### 🧩 Code Quality & Best 
- **HIGH PRIORITY:** Every class, method, and function must include **docstring** explaining its purpose, parameters, and return type.
- Keep components modular and reusable.
- Use **inline comments** to explain complex steps (e.g., TA-Lib calls or regex in cleaning).

### 🧪 Testing
Any script with important logic should have a corresponding test inside `tests/`.

Example:
```bash
scripts/finance_tools.py  <-- tested by --> tests/test_finance_tools.py (Future file)
```