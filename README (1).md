# CEO Sentiment Analysis

> Analyzing how the CEOs of America's top 10 companies communicate about building their businesses — using NLP on SEC EDGAR 10-K filings.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![NLP](https://img.shields.io/badge/NLP-FinBERT%20%7C%20VADER%20%7C%20TextBlob-green?style=flat-square)
![Data Source](https://img.shields.io/badge/Data-SEC%20EDGAR-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

This project builds a complete end-to-end sentiment analysis pipeline that scrapes, cleans, scores, and visualizes CEO shareholder letters from the top 10 US companies by market capitalization. The goal is to quantify how executives communicate about their company's journey, growth, and strategy using three NLP models.

**Companies analyzed:** Apple, Microsoft, Amazon, Nvidia, Alphabet, Meta, Tesla, Berkshire Hathaway, JPMorgan Chase, ExxonMobil

**Data source:** SEC EDGAR 10-K annual filings (2022-2023)

---

## Key Findings

| Metric | Result |
|---|---|
| Average VADER Score | 0.852 (Highly Positive) |
| Average TextBlob Polarity | 0.137 (Slightly Positive) |
| Average FinBERT Score | 0.650 (Positive) |
| Positive Sentiment | 65% of all text chunks |
| Negative Sentiment | 0% across all companies |

### Company Rankings by FinBERT Score

| Rank | Company | CEO | FinBERT Score |
|---|---|---|---|
| 1 | ExxonMobil | Darren Woods | 1.000 |
| 2 | Microsoft | Satya Nadella | 1.000 |
| 3 | Alphabet | Sundar Pichai | 1.000 |
| 4 | Apple | Tim Cook | 1.000 |
| 5 | Nvidia | Jensen Huang | 0.500 |
| 6 | Tesla | Elon Musk | 0.500 |
| 7 | JPMorgan Chase | Jamie Dimon | 0.500 |
| 8 | Meta | Mark Zuckerberg | 0.500 |
| 9 | Berkshire Hathaway | Warren Buffett | 0.500 |
| 10 | Amazon | Andy Jassy | 0.000 |

---

## Project Structure

```
sentiment_analysis/
├── data/
│   ├── raw/                        # Raw scraped CEO text
│   │   └── ceo_comments_raw.csv
│   └── processed/                  # Cleaned, filtered text
│       └── ceo_comments_clean.csv
├── output/
│   ├── charts/                     # All 5 visualizations (PNG)
│   └── results/
│       └── sentiment_results.csv   # Scored output
├── scraper.py                      # SEC EDGAR scraper
├── cleaner.py                      # Text preprocessing
├── sentiment.py                    # VADER + TextBlob + FinBERT scoring
├── visualize.py                    # Chart and report generation
├── main.py                         # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Pipeline Steps

**Step 1 - Scraping**
Scrapes CEO shareholder letters from SEC EDGAR 10-K filings using `requests` and `BeautifulSoup`. Includes random request delays and rotating user-agent headers to avoid blocks. Falls back to sample data with a logged warning if a source is unavailable. Output saved to `data/raw/ceo_comments_raw.csv`.

**Step 2 - Cleaning**
Strips boilerplate legal language, forward-looking statement disclaimers, and financial noise. Chunks long texts into paragraph-level segments and filters by company-building keywords: `built`, `journey`, `founded`, `growth`, `challenge`, `vision`, `strategy`, `milestone`, `decision`, `learned`. Output saved to `data/processed/ceo_comments_clean.csv`.

**Step 3 - Sentiment Analysis**
Runs three models on every cleaned text chunk:
- **VADER** — rule-based, compound score (-1 to +1)
- **TextBlob** — pattern-based polarity (-1 to +1)
- **FinBERT** (`ProsusAI/finbert`) — finance-specific transformer, batched in groups of 32

Output saved to `output/results/sentiment_results.csv`.

**Step 4 - Visualization**
Generates 5 charts saved to `output/charts/`:
1. Bar chart — average FinBERT score per company
2. Line chart — sentiment trend over time per company
3. Heatmap — all three models across all companies
4. Word clouds — positive vs. negative vocabulary
5. Summary metrics table

**Step 5 - Reporting**
Auto-generates a markdown summary report and a formatted Word document overview of all findings.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ceo-sentiment-analysis.git
cd ceo-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```
# Add any API keys here if extending the project
# No keys are required to run the base pipeline
```

### 4. Run the full pipeline

```bash
python main.py
```

All outputs will be saved to the `data/` and `output/` directories automatically.

---

## Requirements

```
requests
beautifulsoup4
selenium
pandas
numpy
nltk
textblob
vaderSentiment
transformers
torch
matplotlib
seaborn
wordcloud
tqdm
python-dotenv
```

Install all at once:

```bash
pip install -r requirements.txt
```

> **Note:** FinBERT requires approximately 440 MB of disk space and benefits from a GPU for faster inference. On CPU, batching in groups of 32 keeps memory usage manageable.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Scraping | Requests, BeautifulSoup4, Selenium |
| NLP | NLTK, TextBlob, HuggingFace Transformers |
| ML Model | ProsusAI/FinBERT |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Data Processing | Pandas, NumPy |
| Reporting | Quarto (HTML), python-docx (Word) |

---

## Insights

Shareholder letters are not neutral updates. They are carefully engineered investor confidence tools. The complete absence of negative sentiment across all 10 companies confirms that executive communications are strategically calibrated to project confidence, regardless of underlying market conditions.

FinBERT consistently outperformed VADER and TextBlob at differentiating between companies, making it the recommended model for any financial NLP work involving SEC filings or formal corporate communications.

---

## Possible Extensions

- Expand the date range to 2018-2025 to capture pre- and post-COVID sentiment shifts
- Include earnings call transcripts for a higher-frequency signal
- Add topic modeling (LDA or BERTopic) to map which themes drive positive language
- Build a real-time monitoring dashboard that alerts on new 10-K filings
- Extend to mid-cap and sector-specific companies for broader benchmarking

---

## License

MIT License. See `LICENSE` for details.

---

*Generated as part of a data science project. March 2026.*
