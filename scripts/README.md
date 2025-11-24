# ⚙️ Scripts Folder

This directory contains **core business logic** for the project, encapsulated in reusable Python classes and functions. This modular approach is key to achieving a professional and maintainable codebase.

## 📘 Purpose
The scripts here support tasks such as:
- **Object-Oriented Design:** Encapsulating related functionality into classes.
- Data loading, cleaning, and transformation logic.
- Reusable components for analysis notebooks.

## 📁 Contents & Core Components
- `__init__.py` — Makes `scripts/` a Python package, enabling direct imports.
- `finance_tools.py` — Contains the **`StockAnalyzer` class** for loading stock data and computing technical indicators.
- `text_processing.py` — Contains the **`NewsCorpusProcessor` class** for text cleaning, corpus preparation, and LDA topic modeling.
- `README.md` — Documentation (You are here).

## 🧩 Code Quality & Best 
- **HIGH PRIORITY:** Every class, method, and function must include **docstring** explaining its purpose, parameters, and return type.
- Keep components modular and reusable.
- Use **inline comments** to explain complex steps (e.g., TA-Lib calls or regex in cleaning).

## 💡 Example Usage (Typical Workflows)
​Here are short examples demonstrating how to use the core classes in a notebook or script:

### ​1. Financial Data Workflow (StockAnalyzer)
```bash
from scripts.finance_tools import StockAnalyzer

# 1. Initialize and load data (uses yfinance)
aapl_analyzer = StockAnalyzer(ticker='AAPL', start='2024-01-01', end='2024-12-01')

# 2. Compute Technical Indicators (modifies the internal DataFrame)
try:
    aapl_analyzer.add_indicators()
    # 3. Access the resulting DataFrame
    print("AAPL Data with Indicators (Head):")
    print(aapl_analyzer.df[['Close', 'RSI_14', 'MACD']].tail())

except ValueError as e:
    print(f"Could not calculate indicators: {e}")
```

### 2. Text Processing Workflow (NewsCorpusProcessor)
```bash
from scripts.text_processing import NewsCorpusProcessor

news_headlines = [
    "Apple stock hits new high after Q3 earnings.",
    "The Federal Reserve announces a rate cut, boosting markets.",
    "Analyst predicts volatile movements next week.",
    "A generic market comment."
]

# 1. Initialize the processor
processor = NewsCorpusProcessor(news_headlines)

# 2. Prepare the corpus (tokenization, cleaning, dictionary creation)
# Use a low filter for small sample data
processor.prepare_corpus(no_below=1, no_above=0.8) 

# 3. Run LDA Topic Modeling
model, topics = processor.lda_topics(num_topics=2, passes=5)

# 4. Print the discovered topics
print("\nDiscovered Topics:")
for topic in topics:
    print(f"- {topic}")
```

## 🧪 Testing
Any script with important logic should have a corresponding test inside `tests/`.

Example:
```bash
scripts/finance_tools.py  <-- tested by --> tests/test_finance_tools.py 
```