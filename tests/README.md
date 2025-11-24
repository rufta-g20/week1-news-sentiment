# 🧪 Tests

This folder contains unit tests for validating the correctness of project modules.

### ✔️ Purpose of Tests
- Ensure data processing functions and **class methods** behave correctly
- Catch errors before merging (via CI)
- Maintain code quality as the project grows

### 📁 Structure
- `__init__.py` — Makes this a Python package
- *(Future)* Add test files such as:
  - `test_finance_tools.py` — For the new **`StockAnalyzer` class**.
  - `test_text_processing.py` — For the new **`NewsCorpusProcessor` class**.

### ▶️ Running Tests Locally
Activate environment and run:

```bash
pytest tests/
```

### 🤖 CI Integration
All tests run automatically via GitHub Actions *(unittests.yml)*:
- On every push
- On every pull request
**Red CI = fix before merging**
**Green CI = safe to merge**