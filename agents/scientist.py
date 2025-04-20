import pandas as pd
import asyncio
import json
import logging
import numpy as np
import re  # Import the re module
import ast
from typing import Dict, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    root_mean_squared_error
)

from utils.gemini_client import generate_insight

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------- INSIGHT HELPERS ----------
def _generate_model_insights(model, features: List[str], target: Optional[str]) -> List[str]:
    insights = []
    try:
        if hasattr(model, 'feature_importances_'):
            imp = dict(zip(features, model.feature_importances_))
            top, val = max(imp.items(), key=lambda x: x[1])
            insights.append(f"Most important feature: '{top}' ({val:.2f})")
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            top, val = max(zip(features, coefs), key=lambda x: abs(x[1]))
            insights.append(f"Strongest predictor: '{top}' ({val:.2f})")
        if target:
            insights.append(f"Model trained to predict '{target}' using {len(features)} features")
    except Exception:
        insights.append("Could not extract model insights")
    return insights


# ---------- GEMINI UTILITIES ----------
async def _safe_generate_insight(prompt: str, retries: int = 2, timeout: float = 10.0) -> str:
    for attempt in range(1, retries + 1):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(generate_insight, prompt), timeout=timeout)
            if isinstance(result, str) and result.strip():
                return result
        except Exception as e:
            logger.warning(f"Gemini error: {e}")
    return ""


def _sanitize_gemini_response(text: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F]', '', text).strip()


def _extract_json_block(text: str) -> Optional[str]:
    m = re.search(r'\{[\s\S]*?\}', text)
    return m.group(0) if m else None


def _validate_code(code: str) -> bool:
    try:
        ast.parse(code)
        blacklist = ['os.', 'sys.', '__import__', 'open(', 'eval(', 'exec(', 'subprocess']
        return not any(bad in code.lower() for bad in blacklist)
    except:
        return False


def _extract_training_code(text: str) -> str:
    match = re.findall(r'```python(?:\s*#\s*\[MODEL TRAINING\])?\n([\s\S]+?)```', text)
    code = "\n".join(match).strip()
    return code if _validate_code(code) else ""


# ---------- EXECUTION ----------
async def _execute_training(df: pd.DataFrame, code: str, config: Dict) -> Dict:
    local_vars = {
        'df': df, 'pd': pd, 'np': np,
        'train_test_split': train_test_split,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'LabelEncoder': LabelEncoder,
        'StandardScaler': StandardScaler,
        'LogisticRegression': LogisticRegression,
        'LinearRegression': LinearRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'SVC': SVC, 'SVR': SVR,
        'classification_report': classification_report,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2_score': r2_score,
        'accuracy_score': accuracy_score,
        'root_mean_squared_error': root_mean_squared_error
    }

    results = {}
    try:
        exec(code, globals(), local_vars)
        model = local_vars.get("model")
        metrics = local_vars.get("metrics", {})

        results["model_type"] = type(model).__name__ if model else "Unknown"
        results["metrics"] = metrics
        if model and config["task"] != "clustering":
            results["insights"] = _generate_model_insights(model, config["feature_columns"], config["target_column"])
    except Exception as e:
        results["warnings"] = [f"Execution warning: {str(e)}"]
        results["metrics"] = {"error": str(e)}
        results["insights"] = ["Failed to generate model insights"]
    return results


# ---------- FALLBACK ----------
async def _enhanced_fallback(df: pd.DataFrame, error: Exception) -> Dict:
    logger.warning(f"Enhanced fallback due to: {error}")
    df2 = df.copy()
    df2 = df2.drop(columns=[c for c in ['id', 'uuid', 'timestamp', 'last_transaction'] if c in df2.columns], errors='ignore')

    # Exclude datetime64 columns from the model
    df2 = df2.select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])

    for c in df2.select_dtypes(include='object').columns:
        df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))

    target = df2.select_dtypes(include='number').var().idxmax()
    features = [c for c in df2.columns if c != target]
    X, y = df2[features], df2[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    task = 'regression' if y.nunique() > 10 else 'classification'
    model = RandomForestRegressor() if task == 'regression' else RandomForestClassifier()
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    metrics = {
        "rmse": root_mean_squared_error(yte, preds),
        "mae": mean_absolute_error(yte, preds),
        "r2": r2_score(yte, preds)
    } if task == 'regression' else {
        "accuracy": accuracy_score(yte, preds),
        "report": classification_report(yte, preds, output_dict=True)
    }

    return {
        "model_type": type(model).__name__,
        "task": task,
        "target": target,
        "features": features,
        "metrics": metrics,
        "insights": [],
        "training_code": None,
        "warnings": ["Used fallback strategy due to Gemini failure"]
    }


# ---------- MAIN ENTRY POINT ----------
async def science(df: pd.DataFrame, analysis_results: Optional[Dict] = None) -> Dict:
    context = analysis_results.get("ai_analysis", {}).get("context", "") if analysis_results else ""
    dtypes = df.dtypes.apply(str).to_dict()
    sample = df.head(3).to_string()

    prompt1 = f"""As an ML engineer, analyze this dataset:
Columns & types: {dtypes}
Sample:\n{sample}
Context: {context}

Identify the ML task (classification, regression, clustering),
choose the target column and feature columns, and justify.
Output JSON:
{{"task":"...","target_column":"...","feature_columns":[...],"rationale":"..."}}"""

    try:
        resp1 = await _safe_generate_insight(prompt1)
        json_str = _extract_json_block(resp1)
        if not json_str:
            raise ValueError("Gemini did not return JSON")

        config = json.loads(_sanitize_gemini_response(json_str))
        config['task'] = config['task'].lower()

        # Handle invalid tasks
        if config['task'] not in ['classification', 'regression', 'clustering']:
            logger.warning(f"Invalid task '{config['task']}' received from Gemini. Defaulting to regression.")
            config['task'] = 'regression'

        if config['target_column'] not in df.columns:
            raise ValueError(f"Invalid target: {config['target_column']}")
        invalid = [col for col in config['feature_columns'] if col not in df.columns]
        if invalid:
            raise ValueError(f"Invalid feature columns: {invalid}")

        prompt2 = f"""Write complete scikit-learn training code for:
Task: {config['task']}
Target: {config['target_column']}
Features: {config['feature_columns']}
Use train_test_split, train, predict, and store performance in a `metrics` dict.
End with `model = ...` and `metrics = {{...}}`
Output only:
```python
# [MODEL TRAINING]
...your code...
```"""

        resp2 = await _safe_generate_insight(prompt2)
        code = _extract_training_code(resp2)
        if not code:
            raise ValueError("No valid code block extracted from Gemini")

        results = await _execute_training(df, code, config)

        # Generate business insight from model results
        insight_prompt = f"""
You are a lead data scientist. Based on this dataset:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- ML Task: {config['task']}
- Target: {config['target_column']}
- Features used: {config['feature_columns']}
- Model performance: {results.get('metrics')}

Please:
1. Summarize data trends and patterns.
2. Explain what these patterns reveal.
3. Predict future trends or behaviors.
4. Suggest business/modeling recommendations.
"""

        structured_insight = await _safe_generate_insight(insight_prompt)
        if structured_insight.strip():
            results.setdefault("insights", []).append(structured_insight.strip())

        return {
            "model_type": results.get("model_type"),
            "task": config["task"],
            "target": config["target_column"],
            "features": config["feature_columns"],
            "metrics": results.get("metrics"),
            "insights": results.get("insights"),
            "training_code": code,
            "warnings": results.get("warnings", [])
        }

    except Exception as e:
        logger.error(f"AI path failed: {e}")
        return await _enhanced_fallback(df, e)
