import pandas as pd
import asyncio
import json
import logging
import numpy as np
import re
import ast
from time import sleep
from typing import Dict, List
from utils.gemini_client import generate_insight

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Functions for Missing Value Handling, Code Validation, and Other Utilities
def clean_nans(obj):
    """Recursively clean NaN values from dictionaries and lists"""
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and (pd.isna(obj) or np.isinf(obj)):
        return None
    return obj

def _validate_code(code: str) -> bool:
    """Basic code safety check to avoid harmful operations"""
    try:
        ast.parse(code)
        forbidden = ['os.', 'sys.', '__import__', 'open(', 'eval(', 'exec(', 'subprocess']
        return not any(tok in code.lower() for tok in forbidden)
    except:
        return False

def _get_data_snapshot(df: pd.DataFrame) -> Dict:
    """Returns a snapshot of the data quality: shape, missing values, data types"""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.apply(str).to_dict()
    }

async def _safe_generate_insight(prompt: str, retries: int = 3, timeout: float = 15.0) -> str:
    """Safely generates insights with retry logic"""
    delay = 1
    for attempt in range(retries):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(generate_insight, prompt), timeout=timeout)
            if isinstance(result, str) and result.strip():
                return result
        except Exception as e:
            logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
        sleep(delay)
        delay *= 2
    return ""

async def _execute_critical_fixes(df: pd.DataFrame, analysis: Dict) -> List[Dict]:
    """Auto‑executes high‑priority fixes from analyzer"""
    executed = []
    if not analysis:
        return executed

    fixes = analysis.get('preprocessing_ready', {})
    try:
        # Missing value fixes
        for cmd in fixes.get('missing', []):
            if 'fillna' in cmd or 'dropna' in cmd:
                before = df.isna().sum().to_dict()
                exec(cmd, globals(), {'df': df})
                executed.append({
                    "operation": cmd,
                    "impact": f"Missing values {before} → {df.isna().sum().to_dict()}"
                })

        # Outlier fixes
        for cmd in fixes.get('outliers', []):
            if 'drop' in cmd or 'clip' in cmd:
                before_shape = df.shape
                exec(cmd, globals(), {'df': df})
                executed.append({
                    "operation": cmd,
                    "impact": f"Rows {before_shape[0]} → {df.shape[0]}"
                })

        # Transformation fixes (date conversions, renames, etc.)
        for cmd in fixes.get('transformations', []):
            before_snapshot = _get_data_snapshot(df)
            exec(cmd, globals(), {'df': df, 'pd': pd})
            after_snapshot = _get_data_snapshot(df)
            executed.append({
                "operation": cmd,
                "impact": {
                    "before": before_snapshot,
                    "after": after_snapshot
                }
            })
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        pass

    return executed

async def operate(df: pd.DataFrame, analysis_results: Dict = None) -> Dict:
    """Main function to operate on the data with analysis results"""
    context_str = analysis_results.get('ai_analysis', {}).get('context', "") if analysis_results else ""
    sample_data = df.head(3).to_string()

    # Phase 1: Execute critical fixes from analyzer
    executed_ops = await _execute_critical_fixes(df, analysis_results)

    # Phase 2: Get domain-specific suggestions and execute them
    prompt = f"""As a data engineer, suggest pandas operations for:
- Data cleaning
- Feature engineering
- Quality improvement

Business Context:
{context_str}

Current Data Sample:
{sample_data}

Provide 3–5 operations with executable code blocks formatted as:
```python
# [Purpose]
[code]
```"""

    try:
        suggestions = await _safe_generate_insight(prompt)
        suggested_ops = _parse_code_blocks(suggestions)

        # Execute the suggested operations
        for op in suggested_ops:
            if op['safe_to_execute']:
                try:
                    exec(op['code'], globals(), {'df': df})
                    executed_ops.append({
                        "operation": op['code'],
                        "impact": f"Executed: {op['purpose']}"
                    })
                except Exception as e:
                    executed_ops.append({
                        "operation": op['code'],
                        "error": str(e)
                    })

        return clean_nans({
            "executed_operations": executed_ops,
            "suggested_operations": suggested_ops,
            "data_snapshot": _get_data_snapshot(df),
            "processed_df": df
        })

    except Exception as e:
        return {
            "error": str(e),
            "executed_operations": executed_ops,
            "fallback_suggestions": [
                "df.fillna(method='ffill')",
                "df = pd.get_dummies(df, columns=['category'])"
            ]
        }

def _parse_code_blocks(text: str) -> List[Dict]:
    """Extracts executable code from Gemini response"""
    blocks = []
    pattern = r'```python\n# (.+?)\n(.+?)\n```'
    for match in re.finditer(pattern, text, re.DOTALL):
        code = match.group(2).strip()
        blocks.append({
            "purpose": match.group(1).strip(),
            "code": code,
            "safe_to_execute": _validate_code(code)
        })
    return blocks
