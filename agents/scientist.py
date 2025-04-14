# agents/scientist.py
from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict

async def science(df: pd.DataFrame) -> Dict:
    """Generate advanced insights and patterns"""
    sample_data = df.describe().to_string()
    
    prompt = f"""As a data scientist, analyze this dataset and provide:
    1. Key trends and patterns
    2. Interesting correlations
    3. Potential ML applications
    4. Any data anomalies
    
    Statistics Summary:
    {sample_data}
    """
    
    try:
        insights = generate_insight(prompt)
        return {
            "statistical_summary": sample_data,
            "ai_insights": insights
        }
    except Exception as e:
        return {"error": str(e)}