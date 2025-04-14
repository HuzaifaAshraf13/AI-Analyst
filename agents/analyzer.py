# agents/analyzer.py
from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict

async def analyze(df: pd.DataFrame) -> Dict:
    """AI-powered comprehensive data profiling"""
    sample_data = f"""
    Dataset Sample (first 5 rows):
    {df.head().to_string()}
    
    Dataset Shape: {df.shape}
    """
    
    prompt = f"""Act as a senior data analyst. Provide detailed analysis of this dataset:
    1. Data Structure Analysis
    2. Data Quality Assessment
    3. Statistical Overview
    4. Potential Data Issues
    5. Recommendations for Cleaning
    
    {sample_data}"""
    
    try:
        ai_analysis = generate_insight(prompt)
        return {
            "basic_stats": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": str(df.dtypes.to_dict()),
                "missing_values": df.isna().sum().to_dict()
            },
            "ai_analysis": ai_analysis
        }
    except Exception as e:
        return {"error": str(e)}