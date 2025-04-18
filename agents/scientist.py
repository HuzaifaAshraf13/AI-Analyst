import pandas as pd
from typing import Dict, Optional, List
import ast
import re
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from utils.gemini_client import generate_insight

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def science(df: pd.DataFrame, analysis_results: Optional[Dict] = None) -> Dict:
    """
    AI-powered automatic model building with:
    1. Task detection (classification/regression/clustering)
    2. Algorithm selection
    3. Full training pipeline generation
    4. Evaluation and interpretation
    Falls back to a simple RandomForest if Gemini codegen fails.
    """
    context = ""
    if analysis_results:
        ai_analysis = analysis_results.get('ai_analysis', {})
        context = ai_analysis.get('context', "")

    sample_data = df.head(3).to_string()
    dtypes = df.dtypes.apply(str).to_dict()

    task_prompt = f"""As an ML engineer, analyze this dataset:
    Columns and types: {dtypes}

    Sample Data:
    {sample_data}

    Business Context:
    {context}

    Please ensure:
    - Accurately identify if the task is **classification**, **regression**, or **clustering**.
    - Suggest the most relevant **target column** and **features** based on domain knowledge.
    - Explain **why** these features are the most predictive for the target variable.

    Output in JSON format:
    {{
        "task": "classification|regression|clustering",
        "target_column": "column_name|None",
        "feature_columns": ["col1", "col2"],
        "rationale": "Your detailed reasoning on task and features."
    }}"""

    try:
        task_json = generate_insight(task_prompt)
        if not task_json:
            raise ValueError("Empty response from Gemini AI")

        logger.debug(f"Raw Gemini JSON response:\n{task_json}")

        # Clean and extract the valid JSON from Gemini's response
        m = re.search(r'\{[\s\S]*\}', task_json)
        task_json_clean = m.group(0) if m else task_json

        task_config = json.loads(task_json_clean)

        if 'task' not in task_config or 'target_column' not in task_config or 'feature_columns' not in task_config:
            raise ValueError("Invalid task configuration from Gemini")

        # Phase 2: Ask Gemini to generate training code
        code_prompt = f"""Create complete scikit-learn code for:
        Task: {task_config['task']}
        Target: {task_config.get('target_column', 'None')}
        Features: {task_config['feature_columns']}

        Include:
        1. Train-test split (80-20 split)
        2. Model initialization
        3. Training
        4. Evaluation metrics
        5. Feature importance

        Format as:
        ```python
        # [MODEL TRAINING]
        [code]
        ```"""
        
        response = generate_insight(code_prompt)
        training_code = _extract_training_code(response)
        if not training_code:
            raise ValueError("No valid training code extracted")

        # Phase 3: Execute and evaluate
        results = await _execute_training(df, training_code, task_config)

        return {
            "model_type": results.get("model_type"),
            "task": task_config["task"],
            "target": task_config.get("target_column"),
            "features": task_config["feature_columns"],
            "metrics": results.get("metrics", {}),
            "insights": results.get("insights", []),
            "training_code": training_code,
            "warnings": results.get("warnings", [])
        }

    except Exception as e:
        logger.error(f"Error in AI-driven path, falling back: {e}")

        # BUILT-IN FALLBACK: RandomForest
        return await _fallback_random_forest(df, e)


async def _fallback_random_forest(df: pd.DataFrame, error: Exception) -> Dict:
    """Handles fallback to RandomForest in case of failure"""
    logger.warning(f"Fallback triggered due to: {str(error)}")

    # Drop irrelevant columns
    drop_cols = ['id', 'name', 'currency', 'last_transaction']
    df_fb = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Label-encode any remaining object columns
    for col in df_fb.select_dtypes(include='object').columns:
        df_fb[col] = LabelEncoder().fit_transform(df_fb[col])

    numerics = df_fb.select_dtypes(include='number').columns.tolist()
    target = numerics[-1] if numerics else df.columns[-1]

    # Split data into features and target
    X, y = df_fb.drop(columns=[target]), df_fb[target]

    if X.empty or y.empty:
        raise ValueError("Fallback training data is empty or invalid.")

    # Determine if the task is regression or classification
    task_type = 'regression' if y.nunique() > 10 else 'classification'

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model based on task type
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
            "accuracy": (preds == y_test).mean(),
            "classification_report": classification_report(y_test, preds, output_dict=True)
        }

    insights = [f"Fallback {task_type} model trained on target '{target}'"]
    return {
        "model_type": type(model).__name__,
        "task": task_type,
        "target": target,
        "features": list(X.columns),
        "metrics": metrics,
        "insights": insights,
        "training_code": None,
        "warnings": [f"Fallback used due to: {str(error)}"]
    }


async def _execute_training(df: pd.DataFrame, code: str, config: Dict) -> Dict:
    """Safely executes the generated training code and captures model & metrics"""
    results: Dict = {}
    local_vars = {
        'df': df,
        'pd': pd,
        'train_test_split': train_test_split
    }
    try:
        exec(code, globals(), local_vars)
        model = local_vars.get('model')
        results["model_type"] = type(model).__name__ if model is not None else "Unknown"
        results["metrics"] = local_vars.get('metrics', {})
        if config.get('task') != 'clustering' and model is not None:
            results["insights"] = _generate_model_insights(
                model,
                config.get('feature_columns', []),
                config.get('target_column')
            )
    except Exception as ex:
        results["warnings"] = [f"Execution warning: {str(ex)}"]
    return results


def _extract_training_code(text: str) -> str:
    """Extracts the model training code block from Gemini's response."""
    logger.debug("Raw response from Gemini:\n%s", text)

    # Try matching explicitly labeled model training code
    match = re.search(r'```python\s*#\s*\[MODEL TRAINING\]\s*(.+?)```', text, re.DOTALL)
    
    if not match:
        # Fallback: match any python code block
        match = re.search(r'```python\s*(.+?)```', text, re.DOTALL)
    
    if not match:
        logger.warning("No valid code block found in Gemini response.")
        return ""

    code = match.group(1).strip()

    if not _validate_code(code):
        logger.warning("Extracted code block is not valid Python.")
        return ""

    logger.debug("Extracted training code:\n%s", code)
    return code


def _validate_code(code: str) -> bool:
    """Basic safety check for generated code"""
    try:
        ast.parse(code)
        forbidden = ['os.', 'sys.', 'exec(', 'eval(', 'subprocess', 'pickle.load']
        return not any(tok in code.lower() for tok in forbidden)
    except:
        return False


def _generate_model_insights(
    model,
    features: List[str],
    target: Optional[str]
) -> List[str]:
    """Extracts human-readable insights from the trained model"""
    insights: List[str] = []
    try:
        if hasattr(model, 'feature_importances_'):
            imp = dict(zip(features, model.feature_importances_))
            top_f, top_v = max(imp.items(), key=lambda x: x[1])
            insights.append(f"Most important feature: '{top_f}' (importance: {top_v:.2f})")
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            pairs = dict(zip(features, coefs))
            top_f, top_v = max(pairs.items(), key=lambda x: abs(x[1]))
            insights.append(f"Strongest predictor: '{top_f}' (coef: {top_v:.2f})")
        if target:
            insights.append(f"Model trained to predict '{target}' using {len(features)} features")
    except Exception:
        insights.append("Could not extract detailed model insights")
    return insights
