from utils.gemini_client import generate_insight
import pandas as pd
from typing import Dict, List
import ast
import re

async def operate(df: pd.DataFrame, analysis_results: Dict = None) -> Dict:
    """
    AI-driven data operations with context awareness
    Performs 3 key functions:
    1. Executes analyzer's recommended preprocessing
    2. Suggests domain-specific enhancements
    3. Implements critical fixes automatically
    """
    # Pull the raw BUSINESS CONTEXT string out of the analyzer result
    context_str = (
        analysis_results
        .get('ai_analysis', {})
        .get('context', "")
        if analysis_results else ""
    )
    sample_data = df.head(3).to_string()

    # Phase 1: Execute critical fixes from analyzer
    executed_ops = await _execute_critical_fixes(df, analysis_results)

    # Phase 2: Get domain-specific suggestions
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
        suggestions = generate_insight(prompt)
        return {
            "executed_operations": executed_ops,
            "suggested_operations": _parse_code_blocks(suggestions),
            "data_snapshot": _get_data_snapshot(df)
        }
    except Exception as e:
        return {
            "error": str(e),
            "executed_operations": executed_ops,
            "fallback_suggestions": [
                "df.fillna(method='ffill')",
                "df = pd.get_dummies(df, columns=['category'])"
            ]
        }

async def _execute_critical_fixes(df: pd.DataFrame, analysis: Dict) -> List[Dict]:
    """Auto‑executes high‑priority fixes from analyzer"""
    executed = []
    if not analysis:
        return executed

    fixes = analysis.get('preprocessing_ready', {})
    try:
        # Missing‑value fixes
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

        # Transformation fixes (e.g., date conversions, renames, drops)
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

    except Exception:
        # swallow errors so we can still return partial executions
        pass

    return executed

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

def _validate_code(code: str) -> bool:
    """Basic code safety check"""
    try:
        ast.parse(code)
        forbidden = ['os.', 'sys.', 'exec(', 'eval(']
        return not any(tok in code.lower() for tok in forbidden)
    except:
        return False

def _get_data_snapshot(df: pd.DataFrame) -> Dict:
    """Post-operation quality snapshot"""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.apply(str).to_dict()
    }
