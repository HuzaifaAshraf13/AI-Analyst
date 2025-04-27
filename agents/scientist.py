import pandas as pd
import asyncio
import json
import logging
import numpy as np
import re
import ast
import os
import pickle
from time import sleep
from datetime import datetime
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
    accuracy_score
)

from utils.gemini_client import generate_insight

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
GENERATED_DIR = "generated_training_scripts"
MODELS_DIR = "saved_models"
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Helper: Insights ---
def _generate_model_insights(model, features: List[str], target: Optional[str]) -> List[str]:
    insights = []
    try:
        if hasattr(model, 'feature_importances_'):
            imp = dict(zip(features, model.feature_importances_))
            top, val = max(imp.items(), key=lambda x: x[1])
            if val == 0:
                insights.append("Warning: model feature importances are all zero.")
            else:
                insights.append(f"Most important feature: '{top}' ({val:.2f})")
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            top, val = max(zip(features, coefs), key=lambda x: abs(x[1]))
            insights.append(f"Strongest predictor: '{top}' ({val:.2f})")
        if target:
            insights.append(f"Trained to predict '{target}' using {len(features)} features.")
    except Exception as e:
        insights.append(f"Insight extraction failed: {e}")
    return insights

# --- Safe Gemini call with backoff ---
async def _safe_generate_insight(prompt: str, retries: int = 3, timeout: float = 15.0) -> str:
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

# --- Clean NaNs ---
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

# --- Sanitize and extract ---
def _sanitize(text: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F]', '', text).strip()

def _extract_json(text: str) -> Optional[str]:
    m = re.search(r'\{[\s\S]*?\}', text)
    return m.group(0) if m else None

def _validate_code(code: str) -> bool:
    try:
        ast.parse(code)
        blacklist = ['os.', 'sys.', '__import__', 'open(', 'eval(', 'exec(', 'subprocess']
        return not any(b in code.lower() for b in blacklist)
    except:
        return False

def _extract_code(text: str) -> str:
    match = re.findall(r'```python[\s\S]*?\n([\s\S]+?)```', text)
    code = "\n".join(match).strip()
    return code if _validate_code(code) else ""

# --- Execute training code ---
async def _execute_training(df: pd.DataFrame, code: str, cfg: Dict) -> Dict:
    local = {'df': df.copy()}
    # inject libs
    local.update({
        'train_test_split': train_test_split,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'LogisticRegression': LogisticRegression,
        'LinearRegression': LinearRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'SVC': SVC, 'SVR': SVR,
        'LabelEncoder': LabelEncoder,
        'StandardScaler': StandardScaler,
        'np': np
    })
    out = {}
    try:
        # save script
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(GENERATED_DIR, f"train_{cfg['task']}_{ts}.py")
        with open(path, 'w') as f: f.write(code)
        # exec code
        exec(code, {}, local)
        model = local.get('model')
        metrics = local.get('metrics', {})
        out['model_type'] = type(model).__name__ if model else None
        out['metrics'] = metrics
        out['insights'] = _generate_model_insights(model, cfg['feature_columns'], cfg['target_column'])
        # save model
        if model:
            mp = os.path.join(MODELS_DIR, f"{out['model_type']}_{cfg['target_column']}_{ts}.pkl")
            with open(mp, 'wb') as mf: pickle.dump(model, mf)
            out['model_path'] = mp
    except Exception as e:
        out = {'warnings': [f"Execution error: {e}"]}
    return out

# --- Fallback ---
async def _fallback(df: pd.DataFrame, err: Exception) -> Dict:
    logger.warning(f"Fallback triggered: {err}")
    df2 = df.copy().select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])
    for c in df2.select_dtypes(include='object').columns:
        df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))
    tgt = df2.select_dtypes(include='number').var().idxmax()
    feats = [c for c in df2.columns if c != tgt]
    X, y = df2[feats], df2[tgt]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    task = 'regression' if y.unique().size > 10 else 'classification'
    model = RandomForestRegressor() if task == 'regression' else RandomForestClassifier()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    met = (
        {'rmse': mean_squared_error(yte, pred, squared=False), 'mae': mean_absolute_error(yte, pred), 'r2': r2_score(yte, pred)}
        if task == 'regression' else
        {'accuracy': accuracy_score(yte, pred), 'report': classification_report(yte, pred, output_dict=True)}
    )
    return clean_nans({
        'model_type': type(model).__name__,
        'task': task,
        'target': tgt,
        'features': feats,
        'metrics': met,
        'insights': [],
        'warnings': ['fallback used']
    })

# --- Main ---
async def science(df: pd.DataFrame, analysis_results: Optional[Dict] = None) -> Dict:
    # Steer Gemini to valid targets
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].nunique() <= 20]
    context = (analysis_results.get('ai_analysis', {}) or {}).get('context', '')
    dtypes = df.dtypes.apply(str).to_dict()
    sample = df.head(3).to_string()

    prompt1 = (
        f"As an ML engineer, given columns & types: {dtypes} and sample:\n{sample}\n"
        f"Context: {context}\n\n"
        f"Valid regression targets: {numeric_cols}\n"
        f"Valid classification targets (<=20 classes): {cat_cols}\n\n"
        f"Identify the ML task (classification or regression), choose one valid target_column "
        f"and feature_columns, and provide a rationale. Output JSON:\n"
        f'{{\"task\":\"...\",\"target_column\":\"...\",\"feature_columns\":[...],\"rationale\":\"...\"}}'
    )

    try:
        # Generate the task and feature selection insights from Gemini
        r1 = await _safe_generate_insight(prompt1)
        jb = _extract_json(r1)
        if not jb:
            raise ValueError("No JSON from Gemini")
        cfg = json.loads(_sanitize(jb))
        cfg['task'] = cfg.get('task', 'regression').lower()
        if cfg['task'] not in ['classification', 'regression']:
            cfg['task'] = 'regression'

        # Validate target column dtype
        tgt = cfg['target_column']
        if df[tgt].nunique() <= 1:
            raise ValueError(f"Target variable '{tgt}' has constant values. R² cannot be computed.")

        if cfg['task'] == 'regression' and tgt not in numeric_cols:
            raise ValueError(f"Invalid regression target: {tgt}")
        if cfg['task'] == 'classification' and tgt not in cat_cols:
            raise ValueError(f"Invalid classification target: {tgt}")

        # Validate features
        invalid = [c for c in cfg.get('feature_columns', []) if c not in df.columns]
        if invalid:
            raise ValueError(f"Invalid feature columns: {invalid}")

        # Split the data before preprocessing
        X = df[cfg['feature_columns']]
        y = df[tgt]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Generate model training code using Gemini
        prompt2 = (
            f"Write complete scikit-learn training code for:\n"
            f"Task: {cfg['task']}\n"
            f"Target: {tgt}\n"
            f"Features: {cfg['feature_columns']}\n"
            f"Use train_test_split, train, predict, and store performance in a `metrics` dict.\n"
            f"End with `model = ...` and `metrics = {{...}}`\n"
            f"Output only:\n```python\n# [MODEL TRAINING]\n...code...\n```"
        )

        r2 = await _safe_generate_insight(prompt2)
        code = _extract_code(r2)
        if not code:
            raise ValueError("No valid code extracted")

        res = await _execute_training(df, code, cfg)

        # Business insights based on the generated metrics
        bp = f"Given performance {res.get('metrics')}, summarize key trends and recommendations."
        bi = await _safe_generate_insight(bp)
        if bi:
            res.setdefault('insights', []).append(bi.strip())

        # Return results with insights
        return clean_nans({**cfg, **res})

    except Exception as e:
        logger.error(f"Error during science execution: {e}")
        return await _fallback(df, e)

