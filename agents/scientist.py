import pandas as pd
from typing import Dict, Optional, List
import ast
import re
import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from utils.gemini_client import generate_insight

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def science(df: pd.DataFrame, analysis_results: Optional[Dict] = None) -> Dict:
    context = analysis_results.get("ai_analysis", {}).get("context", "") if analysis_results else ""
    sample_data = df.head(3).to_string()
    dtypes = df.dtypes.apply(str).to_dict()

    task_prompt = f"""As an ML engineer, analyze this dataset:
Columns and types: {dtypes}

Sample Data:
{sample_data}

Business Context:
{context}

Please ensure:
- Identify the ML task (classification, regression, or clustering)
- Suggest the most suitable target column and feature columns
- Justify the task choice

Output JSON:
{{
  "task": "classification|regression|clustering",
  "target_column": "column_name",
  "feature_columns": ["col1", "col2"],
  "rationale": "reason"
}}"""

    try:
        task_json = generate_insight(task_prompt)
        if not task_json:
            raise ValueError("Empty response from Gemini")

        logger.debug("Raw Gemini JSON:\n%s", task_json)
        task_json_clean = _sanitize_gemini_response(task_json)
        task_config = json.loads(task_json_clean)

        if not all(k in task_config for k in ['task', 'target_column', 'feature_columns']):
            raise ValueError("Gemini task config missing required keys")

        code_prompt = f"""Write complete scikit-learn training code for:
Task: {task_config['task']}
Target: {task_config['target_column']}
Features: {task_config['feature_columns']}

Include:
- Train-test split (80-20)
- Model training
- Metrics
- Feature importance

Format like:
```python
# [MODEL TRAINING]
[code]
```"""
        code_response = generate_insight(code_prompt)
        training_code = _extract_training_code(code_response)

        if not training_code:
            raise ValueError("No training code extracted")

        results = await _execute_training(df, training_code, task_config)

        return {
            "model_type": results.get("model_type"),
            "task": task_config["task"],
            "target": task_config["target_column"],
            "features": task_config["feature_columns"],
            "metrics": results.get("metrics", {}),
            "insights": results.get("insights", []),
            "training_code": training_code,
            "warnings": results.get("warnings", [])
        }

    except Exception as e:
        logger.error(f"Error in AI-driven path, falling back: {e}")
        return await _fallback_random_forest(df, e)

# ------------------ Fallback Logic ------------------

async def _fallback_random_forest(df: pd.DataFrame, error: Exception) -> Dict:
    logger.warning(f"Fallback triggered due to: {error}")

    df_fb = df.drop(columns=[c for c in ['id', 'name', 'currency', 'last_transaction'] if c in df.columns], errors='ignore')

    for col in df_fb.select_dtypes(include='object').columns:
        df_fb[col] = LabelEncoder().fit_transform(df_fb[col])

    numerics = df_fb.select_dtypes(include='number').columns.tolist()
    target = numerics[-1] if numerics else df.columns[-1]

    X, y = df_fb.drop(columns=[target]), df_fb[target]

    if y.isna().any():
        logger.warning("Target contains NaNs, filling...")
        y = y.fillna(y.median())

    if X.empty or y.empty:
        raise ValueError("Fallback training data is invalid")

    task_type = "regression" if y.nunique() > 10 else "classification"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "regression":
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "rmse": mean_squared_error(y_test, preds, squared=False),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "classification_report": classification_report(y_test, preds, output_dict=True)
        }

    return {
        "model_type": type(model).__name__,
        "task": task_type,
        "target": target,
        "features": list(X.columns),
        "metrics": metrics,
        "insights": [f"Fallback {task_type} model trained on target '{target}'"],
        "training_code": None,
        "warnings": [f"Fallback used due to: {str(error)}"]
    }

# ------------------ Utility Helpers ------------------

async def _execute_training(df: pd.DataFrame, code: str, config: Dict) -> Dict:
    local_vars = {
        'df': df,
        'pd': pd,
        'np': np,
        'train_test_split': train_test_split,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'LabelEncoder': LabelEncoder,
        'StandardScaler': StandardScaler,
        'LogisticRegression': LogisticRegression,
        'LinearRegression': LinearRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'SVC': SVC,
        'SVR': SVR,
        'classification_report': classification_report,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2_score': r2_score,
        'accuracy_score': accuracy_score
    }

    results = {}
    try:
        exec(code, globals(), local_vars)
        model = local_vars.get('model')
        results["model_type"] = type(model).__name__ if model else "Unknown"
        results["metrics"] = local_vars.get("metrics", {})
        if model and config["task"] != "clustering":
            results["insights"] = _generate_model_insights(
                model,
                config["feature_columns"],
                config.get("target_column")
            )
    except Exception as e:
        results["warnings"] = [f"Execution warning: {str(e)}"]
    return results


def _extract_training_code(text: str) -> str:
    logger.debug("Raw response:\n%s", text)
    match = re.search(r'```python\s*#\s*\[MODEL TRAINING\]\s*(.+?)```', text, re.DOTALL)
    if not match:
        match = re.search(r'```python\s*(.+?)```', text, re.DOTALL)
    if not match:
        logger.warning("No valid Python code found in Gemini response.")
        return ""
    code = match.group(1).strip()
    return code if _validate_code(code) else ""


def _validate_code(code: str) -> bool:
    try:
        ast.parse(code)
        return not any(bad in code.lower() for bad in ['os.', 'sys.', 'exec(', 'eval(', 'subprocess', 'pickle'])
    except:
        return False


def _generate_model_insights(model, features: List[str], target: Optional[str]) -> List[str]:
    insights = []
    try:
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(features, model.feature_importances_))
            top_f, top_v = max(imp.items(), key=lambda x: x[1])
            insights.append(f"Most important feature: '{top_f}' (importance: {top_v:.2f})")
        elif hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            top_f, top_v = max(zip(features, coefs), key=lambda x: abs(x[1]))
            insights.append(f"Strongest predictor: '{top_f}' (coef: {top_v:.2f})")
        if target:
            insights.append(f"Model trained to predict '{target}' using {len(features)} features")
    except Exception:
        insights.append("Model insight extraction failed.")
    return insights


def _sanitize_gemini_response(response: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F]', '', response).strip()
