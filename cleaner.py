"""
Text Cleaner for CEO Comments
Cleans and preprocesses scraped text, removing boilerplate and filtering relevant content
"""

import re
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keywords indicating CEO reflections on company building
RELEVANT_KEYWORDS = [
    'built', 'journey', 'founded', 'growth', 'challenge', 'vision',
    'strategy', 'milestone', 'decision', 'learned', 'create', 'develop',
    'establish', 'transform', 'innovate', 'evolve', 'succeed', 'achieve',
    'accomplish', 'overcome', 'navigate', 'lead', 'pioneer', 'mission'
]

# Boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    # Forward-looking statements
    r'(?i)forward[- ]looking\s+statement[s]?.*?(?=\n\n|\Z)',
    r'(?i)this\s+(report|document|filing)\s+contains\s+forward[- ]looking.*?(?=\n\n|\Z)',
    r'(?i)statements\s+regarding\s+future.*?(?=\n\n|\Z)',
    r'(?i)we\s+caution\s+readers.*?(?=\n\n|\Z)',
    r'(?i)actual\s+results\s+may\s+differ\s+materially.*?(?=\n\n|\Z)',
    
    # Legal disclaimers
    r'(?i)safe\s+harbor\s+statement.*?(?=\n\n|\Z)',
    r'(?i)risk\s+factors.*?(?=\n\n|\Z)',
    r'(?i)important\s+(legal\s+)?notice.*?(?=\n\n|\Z)',
    r'(?i)legal\s+disclaimer.*?(?=\n\n|\Z)',
    
    # Financial disclaimers
    r'(?i)non[- ]?gaap\s+financial\s+measures.*?(?=\n\n|\Z)',
    r'(?i)reconciliation\s+of\s+non[- ]?gaap.*?(?=\n\n|\Z)',
    r'(?i)this\s+information\s+should\s+be\s+read\s+in\s+conjunction.*?(?=\n\n|\Z)',
    
    # SEC filing boilerplate
    r'(?i)form\s+10[- ]?k.*?annual\s+report.*?(?=\n\n|\Z)',
    r'(?i)securities\s+and\s+exchange\s+commission.*?(?=\n\n|\Z)',
    r'(?i)pursuant\s+to\s+section\s+\d+.*?(?=\n\n|\Z)',
    r'(?i)exhibit\s+\d+.*?(?=\n\n|\Z)',
    
    # Table of contents, page numbers
    r'(?i)table\s+of\s+contents.*?(?=\n\n|\Z)',
    r'\n\s*\d+\s*\n',  # Standalone page numbers
    r'(?i)part\s+[ivxl]+\s*(?:item\s+\d+)?',  # Part I, Part II, etc.
    
    # Copyright and trademarks
    r'(?i)©\s*\d{4}.*?(?=\n|\Z)',
    r'(?i)all\s+rights\s+reserved.*?(?=\n|\Z)',
    r'®|™|©',
]

# Phrases indicating boilerplate sections to skip entirely
SKIP_SECTION_INDICATORS = [
    'item 1a. risk factors',
    'item 1b. unresolved staff comments',
    'item 7a. quantitative and qualitative disclosures',
    'signatures',
    'power of attorney',
    'certifications',
    'index to financial statements',
    'notes to consolidated financial statements',
]


class TextCleaner:
    """Cleans and preprocesses CEO comment text"""
    
    def __init__(self, input_file: str = "data/raw/ceo_comments_raw.csv",
                 output_dir: str = "data/processed"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_data: List[Dict] = []
        
    def remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate legal and regulatory language"""
        cleaned = text
        
        # Apply regex patterns to remove boilerplate
        for pattern in BOILERPLATE_PATTERNS:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.DOTALL)
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = re.sub(r'\t+', ' ', cleaned)
        
        return cleaned.strip()
    
    def chunk_into_paragraphs(self, text: str, min_length: int = 100) -> List[str]:
        """Split text into paragraph-level chunks"""
        # Split on double newlines or other paragraph markers
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            # Clean up the paragraph
            para = para.strip()
            para = re.sub(r'\s+', ' ', para)
            
            # Skip short paragraphs
            if len(para) < min_length:
                continue
            
            # Skip paragraphs that are likely section headers
            if len(para.split()) < 10 and para.isupper():
                continue
            
            # Skip boilerplate section indicators
            para_lower = para.lower()
            if any(indicator in para_lower for indicator in SKIP_SECTION_INDICATORS):
                continue
            
            cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def is_relevant(self, text: str) -> bool:
        """Check if text contains relevant keywords about company building journey"""
        text_lower = text.lower()
        
        # Count keyword matches
        keyword_count = sum(1 for keyword in RELEVANT_KEYWORDS if keyword in text_lower)
        
        # Require at least 2 keywords for relevance
        return keyword_count >= 2
    
    def extract_relevant_context(self, text: str, context_size: int = 50) -> str:
        """Extract text around relevant keywords with context"""
        text_lower = text.lower()
        words = text.split()
        relevant_indices = set()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean in RELEVANT_KEYWORDS:
                # Add context around the keyword
                start = max(0, i - context_size // 2)
                end = min(len(words), i + context_size // 2)
                relevant_indices.update(range(start, end))
        
        if not relevant_indices:
            return text
        
        # Extract relevant portions
        sorted_indices = sorted(relevant_indices)
        result_words = [words[i] for i in sorted_indices]
        return ' '.join(result_words)
    
    def clean_record(self, record: Dict) -> List[Dict]:
        """Clean a single record and return chunked paragraphs"""
        raw_text = record.get('raw_text', '')
        
        if not raw_text:
            return []
        
        # Remove boilerplate
        cleaned_text = self.remove_boilerplate(raw_text)
        
        # Chunk into paragraphs
        paragraphs = self.chunk_into_paragraphs(cleaned_text)
        
        # Filter for relevant paragraphs about company journey
        relevant_paragraphs = [p for p in paragraphs if self.is_relevant(p)]
        
        # If no relevant paragraphs found, keep top paragraphs anyway
        if not relevant_paragraphs and paragraphs:
            relevant_paragraphs = paragraphs[:5]  # Keep first 5 paragraphs
            logger.debug(f"No keyword matches for {record['company']}, keeping first paragraphs")
        
        # Create cleaned records for each paragraph
        cleaned_records = []
        for i, para in enumerate(relevant_paragraphs):
            cleaned_records.append({
                'company': record['company'],
                'ceo_name': record['ceo_name'],
                'year': record['year'],
                'source': record['source'],
                'chunk_id': i + 1,
                'cleaned_text': para,
                'word_count': len(para.split()),
                'keyword_matches': sum(1 for kw in RELEVANT_KEYWORDS if kw in para.lower())
            })
        
        return cleaned_records
    
    def clean_all(self) -> List[Dict]:
        """Clean all records from input file"""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return []
        
        # Read input CSV
        df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(df)} raw records from {self.input_file}")
        
        all_cleaned = []
        
        for _, row in df.iterrows():
            record = row.to_dict()
            cleaned_chunks = self.clean_record(record)
            all_cleaned.extend(cleaned_chunks)
            
            if cleaned_chunks:
                logger.debug(f"Cleaned {record['company']} ({record['year']}): {len(cleaned_chunks)} chunks")
        
        self.cleaned_data = all_cleaned
        logger.info(f"Total cleaned chunks: {len(all_cleaned)}")
        
        return all_cleaned
    
    def save_to_csv(self, data: Optional[List[Dict]] = None, filename: str = "ceo_comments_clean.csv"):
        """Save cleaned data to CSV"""
        data = data or self.cleaned_data
        
        if not data:
            logger.warning("No data to save")
            return None
        
        output_path = self.output_dir / filename
        
        fieldnames = ['company', 'ceo_name', 'year', 'source', 'chunk_id', 
                     'cleaned_text', 'word_count', 'keyword_matches']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Saved {len(data)} cleaned records to {output_path}")
        return str(output_path)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the cleaned data"""
        if not self.cleaned_data:
            return {}
        
        df = pd.DataFrame(self.cleaned_data)
        
        stats = {
            'total_chunks': len(df),
            'unique_companies': df['company'].nunique(),
            'unique_years': sorted(df['year'].unique().tolist()),
            'avg_word_count': df['word_count'].mean(),
            'avg_keyword_matches': df['keyword_matches'].mean(),
            'chunks_per_company': df.groupby('company').size().to_dict(),
        }
        
        return stats


def clean_ceo_comments(input_file: str = "data/raw/ceo_comments_raw.csv",
                       output_dir: str = "data/processed") -> str:
    """Main function to clean CEO comments"""
    logger.info("Starting text cleaning and preprocessing...")
    
    cleaner = TextCleaner(input_file=input_file, output_dir=output_dir)
    
    # Clean all records
    cleaned_data = cleaner.clean_all()
    
    if not cleaned_data:
        logger.warning("No cleaned data produced")
        return None
    
    # Save cleaned data
    output_path = cleaner.save_to_csv()
    
    # Log statistics
    stats = cleaner.get_statistics()
    logger.info(f"Cleaning statistics:")
    logger.info(f"  - Total chunks: {stats.get('total_chunks', 0)}")
    logger.info(f"  - Companies: {stats.get('unique_companies', 0)}")
    logger.info(f"  - Years: {stats.get('unique_years', [])}")
    logger.info(f"  - Avg word count: {stats.get('avg_word_count', 0):.1f}")
    
    return output_path


if __name__ == "__main__":
    clean_ceo_comments()
