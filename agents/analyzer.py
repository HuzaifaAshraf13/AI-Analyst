from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict, List, Union
import re
import os


def load_csv(path: str) -> pd.DataFrame:
    """Loads CSV with sane defaults."""
    return pd.read_csv(
        path,
        dtype=str,  # avoid mixed-type errors
        na_values=['', 'NA', 'null'],
        low_memory=False
    )


async def analyze(source: Union[str, pd.DataFrame]) -> Dict:
    """AI-powered data profiling. Accepts a CSV file path or DataFrame."""
    
    # === Step 1: Load data if given a path ===
    if isinstance(source, str) and os.path.isfile(source):
        df = load_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        raise ValueError("Invalid source provided. Must be CSV file path or DataFrame.")
    
    # === Step 2: Prepare prompt ===
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
        print("=== RAW GEMINI OUTPUT ===\n", ai_analysis)

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


# === Helpers ===

def extract_section(text: str, header: str) -> str:
    try:
        headers = re.finditer(r'^\d+\.\s+[A-Z\s]+:', text, re.MULTILINE)
        header_positions = {match.group(): match.start() for match in headers}
        matching_header = next((h for h in header_positions if header.lower() in h.lower()), None)
        if not matching_header:
            return "Section not found"
        start = header_positions[matching_header] + len(matching_header)
        next_headers = [pos for h, pos in header_positions.items() if pos > start]
        end = min(next_headers) if next_headers else len(text)
        return text[start:end].strip()
    except Exception:
        return "Section parsing failed"

def extract_recommendations(text: str) -> List[str]:
    """Extract actionable pandas code from Gemini response"""
    try:
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        recommendations = []

        for block in code_blocks:
            lines = block.strip().split('\n')
            for line in lines:
                clean = line.strip()
                if not clean or clean.startswith("#"):
                    continue
                if any(x in clean for x in ['df[', 'fillna', 'dropna', 'replace', 'clip', 'astype', 'drop']):
                    recommendations.append(clean)
        return recommendations
    except Exception as e:
        print("Recommendation parse error:", e)
        return []



def format_for_operator(recommendations: List[str]) -> Dict:
    priority_map = {
        'missing': [],
        'outliers': [],
        'transformations': []
    }
    missing_keywords = ['missing', 'na', 'null', 'fillna', 'dropna', 'impute']
    outlier_keywords = ['outlier', 'clip', 'remove', 'drop', 'filter', 'iqr', 'zscore']

    for rec in recommendations:
        rec_lower = rec.lower()
        if any(kw in rec_lower for kw in missing_keywords):
            priority_map['missing'].append(rec)
        elif any(kw in rec_lower for kw in outlier_keywords):
            priority_map['outliers'].append(rec)
        else:
            priority_map['transformations'].append(rec)

    return priority_map
