# agents/operator.py
from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict, List

async def operate(df: pd.DataFrame) -> Dict:
    """AI suggests and performs data operations"""
    sample_data = df.head().to_string()
    
    prompt = f"""Suggest 3-5 most valuable pandas operations for this dataset.
    For each suggestion:
    - Explain why it's valuable
    - Provide the exact Python code to execute it
    
    Data Sample:
    {sample_data}
    
    Format your response as:
    1. [Operation Name]: [Brief reason]
    ```python
    [code]
    ```
    """
    
    try:
        operations_suggestion = generate_insight(prompt)
        return {
            "suggested_operations": operations_suggestion,
            "executed_operations": []  # Can be extended to auto-execute
        }
    except Exception as e:
        return {"error": str(e)}