# CEO Sentiment Analysis Pipeline

> **Sentiment analysis on US top company CEO comments from SEC filings**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-FinBERT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## What This Project Does

This automated pipeline analyzes the **tone and sentiment** of CEO shareholder letters from the top 10 US companies by market cap. It scrapes data from SEC EDGAR 10-K filings, cleans the text, and runs triple sentiment analysis using VADER, TextBlob, and FinBERT (a financial domain transformer model).

### Pipeline Steps

1. **Data Collection** - Scrapes CEO letters from SEC EDGAR filings (or uses curated fallback data)
2. **Text Preprocessing** - Removes boilerplate, legal disclaimers, and chunks into paragraphs
3. **Sentiment Analysis** - Runs three models: VADER (rule-based), TextBlob (pattern-based), FinBERT (transformer)
4. **Visualization** - Generates 5 charts and a markdown report
5. **Export** - Saves results to CSV and PNG files

---

## Executive Summary

Analysis of **20 text chunks** from **10 major US companies** across **2022-2023**.

### Overall Sentiment Scores

| Model | Average Score | Scale |
|-------|---------------|-------|
| **VADER** | 0.852 | -1 to 1 |
| **TextBlob** | 0.137 | -1 to 1 |
| **FinBERT** | 0.650 | -1 to 1 |

### FinBERT Sentiment Distribution

- **Positive:** 65.0% (13 chunks)
- **Neutral:** 35.0% (7 chunks)
- **Negative:** 0.0% (0 chunks)

### Company Rankings by CEO Sentiment

| Rank | Company | CEO | Avg FinBERT Score | Dominant Sentiment |
|------|---------|-----|-------------------|-------------------|
| 1 | ExxonMobil | Darren Woods | 1.000 | Positive |
| 2 | Microsoft | Satya Nadella | 1.000 | Positive |
| 3 | Alphabet | Sundar Pichai | 1.000 | Positive |
| 4 | Apple | Tim Cook | 1.000 | Positive |
| 5 | Nvidia | Jensen Huang | 0.500 | Neutral |
| 6 | Tesla | Elon Musk | 0.500 | Neutral |
| 7 | JPMorgan | Jamie Dimon | 0.500 | Neutral |
| 8 | Meta | Mark Zuckerberg | 0.500 | Neutral |
| 9 | Berkshire Hathaway | Warren Buffett | 0.500 | Neutral |
| 10 | Amazon | Andy Jassy | 0.000 | Neutral |

---

## Visualizations

### 1. Sentiment by Company (Bar Chart)
![Sentiment by Company](output/charts/sentiment_by_company.png)

Average FinBERT sentiment score for each company's CEO communications.

### 2. Sentiment Trend Over Time
![Sentiment Trend](output/charts/sentiment_trend.png)

How CEO sentiment has evolved across the analyzed time period.

### 3. Model Comparison Heatmap
![Sentiment Heatmap](output/charts/sentiment_heatmap.png)

Side-by-side comparison of VADER, TextBlob, and FinBERT scores across all companies.

### 4. Word Clouds
![Word Clouds](output/charts/wordcloud_overall.png)

Most frequent words in positive vs. negative sentiment text chunks.

### 5. Summary Table
![Summary Table](output/charts/summary_table.png)

Comprehensive metrics table for all companies analyzed.

---

## Installation

```bash
# Clone or navigate to the project
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for VADER)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Usage

### Run Full Pipeline
```bash
python main.py
```

### Run Individual Components
```bash
python scraper.py    # Scrape SEC filings
python cleaner.py    # Clean and preprocess text  
python sentiment.py  # Run sentiment analysis
python visualize.py  # Generate visualizations
```

### Run Interactive Notebook
```bash
quarto render sentiment_analysis.qmd
# Opens sentiment_analysis.html with full analysis
```

---

## Project Structure

```
sentiment_analysis/
├── data/
│   ├── raw/                    # Raw scraped text files
│   └── processed/              # Cleaned text files
├── output/
│   ├── results/
│   │   └── sentiment_results.csv   # All sentiment scores
│   ├── charts/
│   │   ├── sentiment_by_company.png
│   │   ├── sentiment_trend.png
│   │   ├── sentiment_heatmap.png
│   │   ├── wordcloud_overall.png
│   │   └── summary_table.png
│   └── summary_report.md       # Generated markdown report
├── scraper.py                  # SEC EDGAR scraper with fallback
├── cleaner.py                  # Text preprocessing
├── sentiment.py                # Triple sentiment analysis
├── visualize.py                # Chart generation
├── main.py                     # Pipeline orchestrator
├── sentiment_analysis.qmd      # Quarto notebook
├── sentiment_analysis.html     # Rendered HTML report
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Target Companies

| Ticker | Company | CEO | CIK |
|--------|---------|-----|-----|
| AAPL | Apple Inc. | Tim Cook | 0000320193 |
| MSFT | Microsoft Corp. | Satya Nadella | 0000789019 |
| AMZN | Amazon.com Inc. | Andy Jassy | 0001018724 |
| NVDA | NVIDIA Corp. | Jensen Huang | 0001045810 |
| GOOGL | Alphabet Inc. | Sundar Pichai | 0001652044 |
| META | Meta Platforms | Mark Zuckerberg | 0001326801 |
| TSLA | Tesla Inc. | Elon Musk | 0001318605 |
| BRK-A | Berkshire Hathaway | Warren Buffett | 0001067983 |
| JPM | JPMorgan Chase | Jamie Dimon | 0000019617 |
| XOM | ExxonMobil Corp. | Darren Woods | 0000034088 |

---

## Sentiment Models

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type:** Rule-based lexicon
- **Best for:** Social media, short text
- **Output:** Compound score (-1 to 1)
- **Speed:** Very fast

### TextBlob
- **Type:** Pattern-based
- **Best for:** General text
- **Output:** Polarity (-1 to 1) and Subjectivity (0 to 1)
- **Speed:** Fast

### FinBERT (ProsusAI/finbert)
- **Type:** Transformer (BERT fine-tuned)
- **Best for:** Financial text, SEC filings
- **Output:** Positive/Neutral/Negative classification with confidence
- **Speed:** Slower (GPU recommended)

---

## Methodology

### Data Collection
CEO shareholder letters were scraped from SEC EDGAR 10-K filings. The scraper rotates user agents and implements rate limiting to comply with SEC guidelines. Fallback sample data is used when live scraping fails.

### Text Processing
1. Remove boilerplate legal language and forward-looking statement disclaimers
2. Chunk text into paragraph-level segments
3. Filter for paragraphs containing keywords: *built, journey, founded, growth, challenge, vision, strategy, milestone, decision, learned*

### Sentiment Scoring
Each text chunk receives scores from all three models. FinBERT is weighted as the primary model for financial text accuracy.

---

## Output Files

| File | Description |
|------|-------------|
| `output/results/sentiment_results.csv` | Full sentiment scores for all chunks |
| `output/charts/*.png` | 5 visualization charts |
| `output/summary_report.md` | Markdown analysis report |
| `sentiment_analysis.html` | Interactive Quarto HTML report |

---

## Requirements

- Python 3.8+
- ~500MB disk space (for FinBERT model)
- Internet connection (for first-time model download)

See [requirements.txt](requirements.txt) for full dependency list.

---

## License

MIT License

---

*Generated by CEO Sentiment Analysis Pipeline • March 2026*
