"""
Visualization Module
Generates charts and reports from sentiment analysis results
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class SentimentVisualizer:
    """Generates visualizations from sentiment analysis results"""
    
    def __init__(self, input_file: str = "output/results/sentiment_results.csv",
                 output_dir: str = "output/charts"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df: Optional[pd.DataFrame] = None
        self.summary_data: Dict = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load sentiment results data"""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return None
        
        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(self.df)} records from {self.input_file}")
        
        # Convert finbert labels to numeric scores
        self.df['finbert_score'] = self.df['finbert_label'].map({
            'positive': 1.0, 'negative': -1.0, 'neutral': 0.0
        })
        
        return self.df
    
    def create_sentiment_bar_chart(self, filename: str = "sentiment_by_company.png"):
        """Create bar chart of average sentiment per company using FinBERT scores"""
        if self.df is None:
            self.load_data()
        
        # Calculate average FinBERT score per company
        company_sentiment = self.df.groupby('company')['finbert_score'].mean().sort_values(ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map based on sentiment
        colors = ['#d32f2f' if x < -0.2 else '#4caf50' if x > 0.2 else '#ff9800' 
                  for x in company_sentiment.values]
        
        # Create horizontal bar chart
        bars = ax.barh(company_sentiment.index, company_sentiment.values, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, company_sentiment.values):
            x_pos = val + 0.02 if val >= 0 else val - 0.08
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Average FinBERT Sentiment Score', fontsize=12)
        ax.set_ylabel('Company', fontsize=12)
        ax.set_title('Average CEO Sentiment by Company (FinBERT)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlim(-1.2, 1.2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4caf50', label='Positive (> 0.2)'),
            Patch(facecolor='#ff9800', label='Neutral (-0.2 to 0.2)'),
            Patch(facecolor='#d32f2f', label='Negative (< -0.2)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sentiment bar chart to {output_path}")
        return str(output_path)
    
    def create_sentiment_trend_chart(self, filename: str = "sentiment_trend.png"):
        """Create line chart showing sentiment trend over years per company"""
        if self.df is None:
            self.load_data()
        
        # Calculate average sentiment per company per year
        yearly_sentiment = self.df.groupby(['company', 'year'])['finbert_score'].mean().reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Color palette for companies
        companies = yearly_sentiment['company'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(companies)))
        
        for i, company in enumerate(companies):
            company_data = yearly_sentiment[yearly_sentiment['company'] == company]
            ax.plot(company_data['year'], company_data['finbert_score'], 
                   marker='o', linewidth=2, markersize=8, label=company, color=colors[i])
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average FinBERT Sentiment Score', fontsize=12)
        ax.set_title('CEO Sentiment Trends Over Time by Company', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylim(-1.1, 1.1)
        
        # Set x-axis to show integer years
        years = yearly_sentiment['year'].unique()
        ax.set_xticks(sorted(years))
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sentiment trend chart to {output_path}")
        return str(output_path)
    
    def create_sentiment_heatmap(self, filename: str = "sentiment_heatmap.png"):
        """Create heatmap of sentiment scores across all companies and all three models"""
        if self.df is None:
            self.load_data()
        
        # Calculate average scores per company for each model
        company_sentiments = self.df.groupby('company').agg({
            'vader_score': 'mean',
            'textblob_polarity': 'mean',
            'finbert_score': 'mean'
        }).round(3)
        
        # Rename columns for display
        company_sentiments.columns = ['VADER', 'TextBlob', 'FinBERT']
        
        # Sort by average across models
        company_sentiments['avg'] = company_sentiments.mean(axis=1)
        company_sentiments = company_sentiments.sort_values('avg', ascending=False)
        company_sentiments = company_sentiments.drop('avg', axis=1)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        
        sns.heatmap(company_sentiments, annot=True, cmap='RdYlGn', center=0,
                   vmin=-1, vmax=1, fmt='.2f', linewidths=0.5,
                   cbar_kws={'label': 'Sentiment Score'},
                   annot_kws={'size': 12})
        
        ax.set_title('CEO Sentiment Scores: Companies vs Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sentiment Model', fontsize=12)
        ax.set_ylabel('Company', fontsize=12)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sentiment heatmap to {output_path}")
        return str(output_path)
    
    def create_word_clouds(self, filename_prefix: str = "wordcloud"):
        """Create word clouds showing most common positive vs negative words per company"""
        if self.df is None:
            self.load_data()
        
        # Custom stopwords
        stopwords = set(STOPWORDS)
        stopwords.update(['will', 'year', 'also', 'company', 'one', 'may', 'would', 
                         'could', 'including', 'well', 'new', 'see', 'make', 'us',
                         'many', 'every', 's', 'thing', 'way', 'need', 'time'])
        
        output_paths = []
        
        # Create overall positive vs negative word clouds
        positive_texts = ' '.join(
            self.df[self.df['finbert_label'] == 'positive']['cleaned_text'].fillna('')
        )
        negative_texts = ' '.join(
            self.df[self.df['finbert_label'] == 'negative']['cleaned_text'].fillna('')
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Positive word cloud
        if positive_texts.strip():
            wc_positive = WordCloud(
                width=800, height=400, background_color='white',
                colormap='Greens', stopwords=stopwords, max_words=100
            ).generate(positive_texts)
            axes[0].imshow(wc_positive, interpolation='bilinear')
            axes[0].set_title('Positive Sentiment Words', fontsize=14, fontweight='bold', color='green')
        else:
            axes[0].text(0.5, 0.5, 'No positive texts found', ha='center', va='center')
        axes[0].axis('off')
        
        # Negative word cloud
        if negative_texts.strip():
            wc_negative = WordCloud(
                width=800, height=400, background_color='white',
                colormap='Reds', stopwords=stopwords, max_words=100
            ).generate(negative_texts)
            axes[1].imshow(wc_negative, interpolation='bilinear')
            axes[1].set_title('Negative Sentiment Words', fontsize=14, fontweight='bold', color='red')
        else:
            axes[1].text(0.5, 0.5, 'No negative texts found', ha='center', va='center')
        axes[1].axis('off')
        
        plt.suptitle('Word Clouds: Positive vs Negative CEO Statements', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{filename_prefix}_overall.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(str(output_path))
        
        # Create per-company word clouds
        companies = self.df['company'].unique()
        
        fig, axes = plt.subplots(len(companies), 2, figsize=(16, 4 * len(companies)))
        
        for i, company in enumerate(companies):
            company_df = self.df[self.df['company'] == company]
            
            pos_text = ' '.join(company_df[company_df['finbert_label'] == 'positive']['cleaned_text'].fillna(''))
            neg_text = ' '.join(company_df[company_df['finbert_label'] == 'negative']['cleaned_text'].fillna(''))
            
            # Handle case where axes might be 1D if there's only one company
            if len(companies) == 1:
                ax_pos, ax_neg = axes[0], axes[1]
            else:
                ax_pos, ax_neg = axes[i, 0], axes[i, 1]
            
            # Positive
            if pos_text.strip():
                try:
                    wc = WordCloud(width=400, height=200, background_color='white',
                                 colormap='Greens', stopwords=stopwords, max_words=50).generate(pos_text)
                    ax_pos.imshow(wc, interpolation='bilinear')
                except:
                    ax_pos.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            else:
                ax_pos.text(0.5, 0.5, 'No positive texts', ha='center', va='center')
            ax_pos.set_title(f'{company} - Positive', fontsize=10)
            ax_pos.axis('off')
            
            # Negative
            if neg_text.strip():
                try:
                    wc = WordCloud(width=400, height=200, background_color='white',
                                 colormap='Reds', stopwords=stopwords, max_words=50).generate(neg_text)
                    ax_neg.imshow(wc, interpolation='bilinear')
                except:
                    ax_neg.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            else:
                ax_neg.text(0.5, 0.5, 'No negative texts', ha='center', va='center')
            ax_neg.set_title(f'{company} - Negative', fontsize=10)
            ax_neg.axis('off')
        
        plt.suptitle('Word Clouds by Company: Positive vs Negative', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{filename_prefix}_by_company.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(str(output_path))
        
        logger.info(f"Saved word clouds to {output_paths}")
        return output_paths
    
    def create_summary_table(self, filename: str = "summary_table.png"):
        """Create summary table showing CEO name, company, dominant sentiment, and average score"""
        if self.df is None:
            self.load_data()
        
        # Calculate summary statistics per company
        summary_data = []
        
        for company in self.df['company'].unique():
            company_df = self.df[self.df['company'] == company]
            
            summary_data.append({
                'Company': company,
                'CEO': company_df['ceo_name'].iloc[0],
                'Dominant Sentiment': company_df['finbert_label'].mode().iloc[0],
                'Avg VADER': company_df['vader_score'].mean(),
                'Avg TextBlob': company_df['textblob_polarity'].mean(),
                'Avg FinBERT': company_df['finbert_score'].mean(),
                'Num Chunks': len(company_df)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Avg FinBERT', ascending=False)
        
        # Round numeric columns
        for col in ['Avg VADER', 'Avg TextBlob', 'Avg FinBERT']:
            summary_df[col] = summary_df[col].round(3)
        
        # Store for report generation
        self.summary_data = summary_df.to_dict('records')
        
        # Create table figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4a86e8'] * len(summary_df.columns)
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Color code sentiment cells
        for i in range(len(summary_df)):
            sentiment = summary_df.iloc[i]['Dominant Sentiment']
            color = '#c8e6c9' if sentiment == 'positive' else '#ffcdd2' if sentiment == 'negative' else '#fff9c4'
            table[(i + 1, 2)].set_facecolor(color)
        
        ax.set_title('CEO Sentiment Summary by Company', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary table to {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, filename: str = "summary_report.md"):
        """Generate written markdown summary report"""
        if self.df is None:
            self.load_data()
        
        # Calculate overall statistics
        total_chunks = len(self.df)
        num_companies = self.df['company'].nunique()
        years = sorted(self.df['year'].unique())
        
        avg_vader = self.df['vader_score'].mean()
        avg_textblob = self.df['textblob_polarity'].mean()
        avg_finbert = self.df['finbert_score'].mean()
        
        finbert_dist = self.df['finbert_label'].value_counts()
        
        # Most positive and negative companies
        company_scores = self.df.groupby('company')['finbert_score'].mean().sort_values()
        most_negative = company_scores.index[0]
        most_positive = company_scores.index[-1]
        
        # Generate report
        report = f"""# CEO Sentiment Analysis Report

## Executive Summary

This report analyzes sentiment in CEO shareholder letters and comments from **{num_companies} major US companies** across yeard **{min(years)} to {max(years)}**. A total of **{total_chunks} text chunks** were analyzed using three sentiment models: VADER, TextBlob, and FinBERT.

## Key Findings

### Overall Sentiment
- **Average VADER Score:** {avg_vader:.3f} (scale: -1 to 1)
- **Average TextBlob Polarity:** {avg_textblob:.3f} (scale: -1 to 1)
- **Average FinBERT Score:** {avg_finbert:.3f} (scale: -1 to 1)

### FinBERT Sentiment Distribution
- **Positive:** {finbert_dist.get('positive', 0)} chunks ({finbert_dist.get('positive', 0)/total_chunks*100:.1f}%)
- **Neutral:** {finbert_dist.get('neutral', 0)} chunks ({finbert_dist.get('neutral', 0)/total_chunks*100:.1f}%)
- **Negative:** {finbert_dist.get('negative', 0)} chunks ({finbert_dist.get('negative', 0)/total_chunks*100:.1f}%)

### Company Rankings
- **Most Positive CEO Sentiment:** {most_positive} (FinBERT avg: {company_scores[most_positive]:.3f})
- **Most Negative CEO Sentiment:** {most_negative} (FinBERT avg: {company_scores[most_negative]:.3f})

## Company-by-Company Analysis

| Company | CEO | Dominant Sentiment | Avg FinBERT Score |
|---------|-----|-------------------|------------------|
"""
        # Add company details
        for company in company_scores.index[::-1]:  # Sort from highest to lowest
            company_df = self.df[self.df['company'] == company]
            ceo = company_df['ceo_name'].iloc[0]
            dominant = company_df['finbert_label'].mode().iloc[0]
            score = company_scores[company]
            report += f"| {company} | {ceo} | {dominant.capitalize()} | {score:.3f} |\n"
        
        report += """
## Methodology

### Data Collection
CEO shareholder letters were scraped from SEC EDGAR 10-K filings. Text was extracted from sections containing executive commentary on company performance and strategy.

### Text Processing
- Removed boilerplate legal language and forward-looking statement disclaimers
- Chunked text into paragraph-level segments
- Filtered for paragraphs containing keywords related to company building journey

### Sentiment Models
1. **VADER**: Rule-based sentiment analyzer optimized for social media text
2. **TextBlob**: Pattern-based sentiment analyzer providing polarity scores
3. **FinBERT**: Transformer model fine-tuned on financial text (ProsusAI/finbert)

## Visualizations

The following charts have been generated:
1. **sentiment_by_company.png** - Bar chart of average FinBERT sentiment by company
2. **sentiment_trend.png** - Line chart of sentiment trends over time
3. **sentiment_heatmap.png** - Heatmap comparing all three models across companies
4. **wordcloud_overall.png** - Word clouds for positive vs negative sentiment
5. **wordcloud_by_company.png** - Per-company word clouds
6. **summary_table.png** - Summary table with all metrics

## Insights & Observations

"""
        # Add insights based on data
        if avg_finbert > 0.2:
            report += "- Overall CEO sentiment is **predominantly positive**, reflecting optimistic communication with shareholders.\n"
        elif avg_finbert < -0.2:
            report += "- Overall CEO sentiment is **notably negative**, which may indicate challenging market conditions or cautious messaging.\n"
        else:
            report += "- Overall CEO sentiment is **neutral to mixed**, balancing optimism with measured caution.\n"
        
        # Model agreement
        vader_textblob_corr = self.df['vader_score'].corr(self.df['textblob_polarity'])
        report += f"- VADER and TextBlob show {'strong' if vader_textblob_corr > 0.7 else 'moderate' if vader_textblob_corr > 0.4 else 'weak'} correlation ({vader_textblob_corr:.2f}), indicating {'consistent' if vader_textblob_corr > 0.5 else 'varying'} sentiment detection.\n"
        
        report += """
## Conclusion

This analysis provides a comprehensive view of CEO communication sentiment across America's largest companies. The combination of rule-based (VADER), pattern-based (TextBlob), and transformer-based (FinBERT) models offers a robust multi-perspective analysis of executive sentiment.

---
*Report generated automatically by the CEO Sentiment Analysis Pipeline*
"""
        
        # Save report
        output_path = Path("output") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved summary report to {output_path}")
        return str(output_path)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all visualizations and report"""
        self.load_data()
        
        outputs = {}
        
        logger.info("Generating sentiment bar chart...")
        outputs['bar_chart'] = self.create_sentiment_bar_chart()
        
        logger.info("Generating sentiment trend chart...")
        outputs['trend_chart'] = self.create_sentiment_trend_chart()
        
        logger.info("Generating sentiment heatmap...")
        outputs['heatmap'] = self.create_sentiment_heatmap()
        
        logger.info("Generating word clouds...")
        outputs['word_clouds'] = self.create_word_clouds()
        
        logger.info("Generating summary table...")
        outputs['summary_table'] = self.create_summary_table()
        
        logger.info("Generating summary report...")
        outputs['report'] = self.generate_summary_report()
        
        return outputs


def visualize_results(input_file: str = "output/results/sentiment_results.csv",
                      output_dir: str = "output/charts") -> Dict[str, str]:
    """Main function to generate all visualizations"""
    logger.info("Starting visualization generation...")
    
    visualizer = SentimentVisualizer(input_file=input_file, output_dir=output_dir)
    outputs = visualizer.generate_all_visualizations()
    
    logger.info(f"All visualizations generated successfully")
    return outputs


if __name__ == "__main__":
    visualize_results()
