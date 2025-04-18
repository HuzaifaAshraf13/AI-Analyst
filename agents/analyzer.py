from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict, List
import re

async def analyze(df: pd.DataFrame) -> Dict:
    """AI-powered comprehensive data profiling with enhanced output structure"""
    sample_data = f"""
    Dataset Sample (first 5 rows):
    {df.head().to_string()}
    
    Dataset Shape: {df.shape}
    Column Types:
    {df.dtypes.to_string()}
    Missing Values:
    {df.isna().sum().to_string()}
    """
    
    prompt = f"""As a senior data analyst, provide this structured analysis:
    
    1. DATA STRUCTURE:
    - Primary key candidates (list column names)
    - Time-series indicators (if any)
    - Categorical/numerical breakdown
    
    2. DATA QUALITY:
    - Missing values summary (specific columns with issues)
    - Outlier detection (specific columns with issues)
    - Data integrity issues
    
    3. BUSINESS CONTEXT:
    - Likely industry domain
    - Potential use cases
    - Key business entities
    
    4. ACTIONABLE RECOMMENDATIONS:
    Provide EXACT pandas operations for:
    - Missing data handling (e.g., "df['age'].fillna(df['age'].median(), inplace=True)")
    - Outlier treatment (e.g., "df = df[df['salary'] < 1000000]")
    - Data transformations (e.g., "df['date'] = pd.to_datetime(df['date'])")
    
    Format each recommendation as a bullet point with executable code.
    
    Dataset:
    {sample_data}"""
    
    try:
        ai_analysis = generate_insight(prompt)
        print("=== RAW GEMINI OUTPUT ===\n", ai_analysis)  # Optional debug

        # Parse the sections from the AI response
        sections = {
            'structure': extract_section(ai_analysis, "1. DATA STRUCTURE:"),
            'quality': extract_section(ai_analysis, "2. DATA QUALITY:"),
            'context': extract_section(ai_analysis, "3. BUSINESS CONTEXT:"),
            'recommendations': extract_recommendations(ai_analysis)
        }
        
        return {
            "technical_profile": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.apply(str).to_dict(),
                "missing_values": df.isna().sum().to_dict(),
                "memory_usage": f"{df.memory_usage(deep=True).sum()/1024:.2f} KB"
            },
            "ai_analysis": sections,
            "preprocessing_ready": format_for_operator(sections['recommendations'])
        }
    except Exception as e:
        return {
            "error": str(e),
            "fallback_stats": {
                "shape": df.shape,
                "columns": list(df.columns),
                "missing_values": df.isna().sum().to_dict()
            }
        }

def extract_section(text: str, header: str) -> str:
    """Extracts a specific section from Gemini's response"""
    try:
        # Find all headers and their positions
        headers = re.finditer(r'^\d+\.\s+[A-Z\s]+:', text, re.MULTILINE)
        header_positions = {match.group(): match.start() for match in headers}
        
        # Allow fuzzy matching for headers
        matching_header = next((h for h in header_positions if header.lower() in h.lower()), None)
        if not matching_header:
            return "Section not found"
            
        start = header_positions[matching_header] + len(matching_header)
        
        # Find the next header or end of text
        next_headers = [pos for h, pos in header_positions.items() if pos > start]
        end = min(next_headers) if next_headers else len(text)
        
        return text[start:end].strip()
    except Exception:
        return "Section parsing failed"

def extract_recommendations(text: str) -> List[str]:
    """Extracts actionable items from recommendations section with code detection"""
    try:
        rec_section = extract_section(text, "4. ACTIONABLE RECOMMENDATIONS:")
        if not rec_section or "not found" in rec_section.lower():
            return []
            
        # Extract both bullet points and code-like patterns
        recommendations = []
        for line in rec_section.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Clean bullet points
            if line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
                
            # Look for code-like patterns
            if ('df[' in line or 'pd.' in line or 
                'fillna' in line or 'drop' in line or 'replace' in line):
                recommendations.append(line)
                
        return recommendations if recommendations else []
    except Exception:
        return []

def format_for_operator(recommendations: List[str]) -> Dict:
    """Structures recommendations for operator.py with better classification"""
    priority_map = {
        'missing': [],
        'outliers': [],
        'transformations': []
    }
    
    # Keywords to identify operation types
    missing_keywords = ['missing', 'na', 'null', 'fillna', 'dropna', 'impute']
    outlier_keywords = ['outlier', 'clip', 'remove', 'drop', 'filter', 'iqr', 'zscore']
    
    for rec in recommendations:
        rec_lower = rec.lower()
        
        # Check for missing data operations
        if any(kw in rec_lower for kw in missing_keywords):
            priority_map['missing'].append(rec)
        # Check for outlier operations
        elif any(kw in rec_lower for kw in outlier_keywords):
            priority_map['outliers'].append(rec)
        else:
            priority_map['transformations'].append(rec)
            
    return priority_map
