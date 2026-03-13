"""
Main Pipeline for CEO Sentiment Analysis
Orchestrates all steps: scraping, cleaning, sentiment analysis, and visualization
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
def setup_logging(log_dir: str = "output") -> logging.Logger:
    """Set up logging with both file and console handlers"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("sentiment_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


class SentimentPipeline:
    """Main pipeline orchestrating all sentiment analysis steps"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = setup_logging(self.config.get('log_dir', 'output'))
        self.results: Dict = {}
        
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'results_dir': 'output/results',
            'charts_dir': 'output/charts',
            'log_dir': 'output',
            'raw_file': 'ceo_comments_raw.csv',
            'clean_file': 'ceo_comments_clean.csv',
            'results_file': 'sentiment_results.csv',
            'use_fallback': True,
            'finbert_batch_size': 32,
        }
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            self.config['raw_data_dir'],
            self.config['processed_data_dir'],
            self.config['results_dir'],
            self.config['charts_dir'],
            self.config['log_dir'],
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        self.logger.debug("All directories verified/created")
    
    def step_scrape(self) -> bool:
        """Step 1: Scrape CEO comments from SEC EDGAR"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: SCRAPING CEO COMMENTS FROM SEC EDGAR")
        self.logger.info("=" * 60)
        
        try:
            from scraper import scrape_ceo_comments
            
            output_path = scrape_ceo_comments(
                output_dir=self.config['raw_data_dir'],
                use_fallback=self.config['use_fallback']
            )
            
            if output_path:
                self.results['scrape'] = {
                    'status': 'success',
                    'output_file': output_path
                }
                self.logger.info(f"Scraping complete: {output_path}")
                return True
            else:
                self.results['scrape'] = {'status': 'failed', 'error': 'No output produced'}
                self.logger.error("Scraping failed: No output file produced")
                return False
                
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            self.results['scrape'] = {'status': 'failed', 'error': error_msg}
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False
    
    def step_clean(self) -> bool:
        """Step 2: Clean and preprocess the scraped data"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: CLEANING AND PREPROCESSING TEXT")
        self.logger.info("=" * 60)
        
        try:
            from cleaner import clean_ceo_comments
            
            input_file = Path(self.config['raw_data_dir']) / self.config['raw_file']
            
            if not input_file.exists():
                self.logger.warning(f"Input file not found: {input_file}")
                # Check if scraping produced a file
                if 'scrape' in self.results and self.results['scrape'].get('output_file'):
                    input_file = Path(self.results['scrape']['output_file'])
                else:
                    self.results['clean'] = {'status': 'failed', 'error': 'No input file'}
                    return False
            
            output_path = clean_ceo_comments(
                input_file=str(input_file),
                output_dir=self.config['processed_data_dir']
            )
            
            if output_path:
                self.results['clean'] = {
                    'status': 'success',
                    'output_file': output_path
                }
                self.logger.info(f"Cleaning complete: {output_path}")
                return True
            else:
                self.results['clean'] = {'status': 'failed', 'error': 'No output produced'}
                self.logger.error("Cleaning failed: No output file produced")
                return False
                
        except Exception as e:
            error_msg = f"Cleaning failed: {str(e)}"
            self.results['clean'] = {'status': 'failed', 'error': error_msg}
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False
    
    def step_analyze(self) -> bool:
        """Step 3: Run sentiment analysis"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: RUNNING SENTIMENT ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            from sentiment import analyze_sentiment
            
            input_file = Path(self.config['processed_data_dir']) / self.config['clean_file']
            
            if not input_file.exists():
                if 'clean' in self.results and self.results['clean'].get('output_file'):
                    input_file = Path(self.results['clean']['output_file'])
                else:
                    self.results['analyze'] = {'status': 'failed', 'error': 'No input file'}
                    return False
            
            output_path = analyze_sentiment(
                input_file=str(input_file),
                output_dir=self.config['results_dir'],
                batch_size=self.config['finbert_batch_size']
            )
            
            if output_path:
                self.results['analyze'] = {
                    'status': 'success',
                    'output_file': output_path
                }
                self.logger.info(f"Sentiment analysis complete: {output_path}")
                return True
            else:
                self.results['analyze'] = {'status': 'failed', 'error': 'No output produced'}
                self.logger.error("Sentiment analysis failed: No output file produced")
                return False
                
        except Exception as e:
            error_msg = f"Sentiment analysis failed: {str(e)}"
            self.results['analyze'] = {'status': 'failed', 'error': error_msg}
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False
    
    def step_visualize(self) -> bool:
        """Step 4: Generate visualizations and report"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: GENERATING VISUALIZATIONS AND REPORT")
        self.logger.info("=" * 60)
        
        try:
            from visualize import visualize_results
            
            input_file = Path(self.config['results_dir']) / self.config['results_file']
            
            if not input_file.exists():
                if 'analyze' in self.results and self.results['analyze'].get('output_file'):
                    input_file = Path(self.results['analyze']['output_file'])
                else:
                    self.results['visualize'] = {'status': 'failed', 'error': 'No input file'}
                    return False
            
            outputs = visualize_results(
                input_file=str(input_file),
                output_dir=self.config['charts_dir']
            )
            
            if outputs:
                self.results['visualize'] = {
                    'status': 'success',
                    'outputs': outputs
                }
                self.logger.info(f"Visualization complete: {len(outputs)} outputs generated")
                return True
            else:
                self.results['visualize'] = {'status': 'failed', 'error': 'No outputs produced'}
                self.logger.error("Visualization failed: No outputs produced")
                return False
                
        except Exception as e:
            error_msg = f"Visualization failed: {str(e)}"
            self.results['visualize'] = {'status': 'failed', 'error': error_msg}
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False
    
    def run(self, steps: Optional[list] = None) -> Dict:
        """Run the complete pipeline or specified steps"""
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("CEO SENTIMENT ANALYSIS PIPELINE")
        self.logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Define available steps
        all_steps = {
            'scrape': self.step_scrape,
            'clean': self.step_clean,
            'analyze': self.step_analyze,
            'visualize': self.step_visualize,
        }
        
        # Determine which steps to run
        steps_to_run = steps if steps else ['scrape', 'clean', 'analyze', 'visualize']
        
        # Track success/failure
        successful_steps = []
        failed_steps = []
        
        for step_name in steps_to_run:
            if step_name not in all_steps:
                self.logger.warning(f"Unknown step: {step_name}")
                continue
            
            try:
                success = all_steps[step_name]()
                if success:
                    successful_steps.append(step_name)
                else:
                    failed_steps.append(step_name)
                    self.logger.warning(f"Step '{step_name}' failed but continuing with next steps...")
            except Exception as e:
                failed_steps.append(step_name)
                self.logger.error(f"Step '{step_name}' raised exception: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Calculate duration
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Log final summary
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Successful steps: {successful_steps}")
        self.logger.info(f"Failed steps: {failed_steps}")
        
        # Build final results
        self.results['summary'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'overall_status': 'success' if not failed_steps else 'partial' if successful_steps else 'failed'
        }
        
        return self.results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CEO Sentiment Analysis Pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['scrape', 'clean', 'analyze', 'visualize'],
                       help='Specific steps to run (default: all)')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Disable fallback to sample data if scraping fails')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for FinBERT processing (default: 32)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'results_dir': 'output/results',
        'charts_dir': 'output/charts',
        'log_dir': 'output',
        'raw_file': 'ceo_comments_raw.csv',
        'clean_file': 'ceo_comments_clean.csv',
        'results_file': 'sentiment_results.csv',
        'use_fallback': not args.no_fallback,
        'finbert_batch_size': args.batch_size,
    }
    
    # Run pipeline
    pipeline = SentimentPipeline(config=config)
    results = pipeline.run(steps=args.steps)
    
    # Print final status
    status = results.get('summary', {}).get('overall_status', 'unknown')
    print(f"\n{'='*60}")
    print(f"Pipeline finished with status: {status.upper()}")
    print(f"{'='*60}")
    
    # Return appropriate exit code
    if status == 'success':
        sys.exit(0)
    elif status == 'partial':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
