"""
Sentiment Analysis Module
Runs VADER, TextBlob, and FinBERT sentiment analysis on cleaned CEO comments
"""

import os
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Multi-model sentiment analyzer using VADER, TextBlob, and FinBERT"""
    
    def __init__(self, input_file: str = "data/processed/ceo_comments_clean.csv",
                 output_dir: str = "output/results",
                 batch_size: int = 32):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Initialize models (lazy loading)
        self._vader = None
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._device = None
        
        self.results: List[Dict] = []
    
    @property
    def vader(self):
        """Lazy load VADER sentiment analyzer"""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                logger.info("VADER initialized successfully")
            except ImportError:
                logger.error("VADER not installed. Install with: pip install vaderSentiment")
                raise
        return self._vader
    
    def _init_finbert(self):
        """Initialize FinBERT model and tokenizer"""
        if self._finbert_model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                logger.info("Loading FinBERT model (this may take a moment)...")
                
                model_name = "ProsusAI/finbert"
                self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Set device
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._finbert_model.to(self._device)
                self._finbert_model.eval()
                
                logger.info(f"FinBERT loaded successfully on {self._device}")
                
            except ImportError:
                logger.error("Transformers/torch not installed. Install with: pip install transformers torch")
                raise
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                raise
    
    def analyze_vader(self, text: str) -> float:
        """Analyze sentiment using VADER, returns compound score (-1 to 1)"""
        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return 0.0
    
    def analyze_textblob(self, text: str) -> float:
        """Analyze sentiment using TextBlob, returns polarity (-1 to 1)"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except ImportError:
            logger.error("TextBlob not installed. Install with: pip install textblob")
            return 0.0
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return 0.0
    
    def analyze_finbert_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Analyze sentiment using FinBERT in batches
        Returns list of (label, confidence) tuples
        """
        import torch
        
        self._init_finbert()
        
        results = []
        
        # FinBERT labels
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        try:
            # Tokenize batch
            inputs = self._finbert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self._finbert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1).values
            
            # Convert to results
            for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
                label = label_map.get(pred, 'neutral')
                results.append((label, float(conf)))
                
        except Exception as e:
            logger.warning(f"FinBERT batch analysis failed: {e}")
            results = [('neutral', 0.5)] * len(texts)
        
        return results
    
    def analyze_finbert_single(self, text: str) -> Tuple[str, float]:
        """Analyze single text with FinBERT"""
        results = self.analyze_finbert_batch([text])
        return results[0] if results else ('neutral', 0.5)
    
    def process_batch_finbert(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Process texts in batches for FinBERT"""
        all_results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self.analyze_finbert_batch(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def analyze_all(self) -> List[Dict]:
        """Run all sentiment models on the cleaned data"""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return []
        
        # Load cleaned data
        df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(df)} cleaned records")
        
        texts = df['cleaned_text'].fillna('').tolist()
        
        # Initialize results with original data
        results = []
        
        # Run VADER on all texts
        logger.info("Running VADER sentiment analysis...")
        vader_scores = []
        for text in tqdm(texts, desc="VADER"):
            vader_scores.append(self.analyze_vader(text))
        
        # Run TextBlob on all texts
        logger.info("Running TextBlob sentiment analysis...")
        textblob_scores = []
        for text in tqdm(texts, desc="TextBlob"):
            textblob_scores.append(self.analyze_textblob(text))
        
        # Run FinBERT in batches
        logger.info(f"Running FinBERT sentiment analysis (batch size: {self.batch_size})...")
        finbert_results = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT batches"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.analyze_finbert_batch(batch)
            finbert_results.extend(batch_results)
        
        # Combine results
        for i, row in df.iterrows():
            finbert_label, finbert_conf = finbert_results[i] if i < len(finbert_results) else ('neutral', 0.5)
            
            result = {
                'company': row['company'],
                'ceo_name': row['ceo_name'],
                'year': row['year'],
                'source': row['source'],
                'chunk_id': row['chunk_id'],
                'cleaned_text': row['cleaned_text'],
                'word_count': row['word_count'],
                'vader_score': round(vader_scores[i], 4),
                'textblob_polarity': round(textblob_scores[i], 4),
                'finbert_label': finbert_label,
                'finbert_confidence': round(finbert_conf, 4),
            }
            results.append(result)
        
        self.results = results
        logger.info(f"Sentiment analysis complete for {len(results)} records")
        
        return results
    
    def save_to_csv(self, data: Optional[List[Dict]] = None, 
                    filename: str = "sentiment_results.csv"):
        """Save sentiment results to CSV"""
        data = data or self.results
        
        if not data:
            logger.warning("No data to save")
            return None
        
        output_path = self.output_dir / filename
        
        fieldnames = ['company', 'ceo_name', 'year', 'source', 'chunk_id',
                     'cleaned_text', 'word_count', 'vader_score', 
                     'textblob_polarity', 'finbert_label', 'finbert_confidence']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Saved {len(data)} sentiment results to {output_path}")
        return str(output_path)
    
    def get_statistics(self) -> Dict:
        """Get statistics about sentiment results"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Convert finbert labels to numeric for averaging
        finbert_numeric = df['finbert_label'].map({
            'positive': 1.0, 'neutral': 0.0, 'negative': -1.0
        })
        
        stats = {
            'total_analyzed': len(df),
            'avg_vader': df['vader_score'].mean(),
            'avg_textblob': df['textblob_polarity'].mean(),
            'avg_finbert_confidence': df['finbert_confidence'].mean(),
            'finbert_distribution': df['finbert_label'].value_counts().to_dict(),
            'sentiment_by_company': {},
        }
        
        # Calculate per-company averages
        for company in df['company'].unique():
            company_df = df[df['company'] == company]
            company_finbert = company_df['finbert_label'].map({
                'positive': 1.0, 'neutral': 0.0, 'negative': -1.0
            })
            
            stats['sentiment_by_company'][company] = {
                'vader_avg': company_df['vader_score'].mean(),
                'textblob_avg': company_df['textblob_polarity'].mean(),
                'finbert_avg': company_finbert.mean(),
                'dominant_sentiment': company_df['finbert_label'].mode().iloc[0] if len(company_df) > 0 else 'neutral'
            }
        
        return stats


def analyze_sentiment(input_file: str = "data/processed/ceo_comments_clean.csv",
                      output_dir: str = "output/results",
                      batch_size: int = 32) -> str:
    """Main function to run sentiment analysis"""
    logger.info("Starting multi-model sentiment analysis...")
    
    analyzer = SentimentAnalyzer(
        input_file=input_file,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    # Run analysis
    results = analyzer.analyze_all()
    
    if not results:
        logger.warning("No sentiment results produced")
        return None
    
    # Save results
    output_path = analyzer.save_to_csv()
    
    # Log statistics
    stats = analyzer.get_statistics()
    
    logger.info("Sentiment Analysis Statistics:")
    logger.info(f"  - Total analyzed: {stats.get('total_analyzed', 0)}")
    logger.info(f"  - Avg VADER score: {stats.get('avg_vader', 0):.3f}")
    logger.info(f"  - Avg TextBlob polarity: {stats.get('avg_textblob', 0):.3f}")
    logger.info(f"  - Avg FinBERT confidence: {stats.get('avg_finbert_confidence', 0):.3f}")
    logger.info(f"  - FinBERT distribution: {stats.get('finbert_distribution', {})}")
    
    return output_path


if __name__ == "__main__":
    analyze_sentiment()
