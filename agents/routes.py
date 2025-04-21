from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import tempfile
import os
from pathlib import Path
import numpy as np

from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science
from agents.reporter import report

router = APIRouter()

def sanitize_for_json(data):
    """Recursively clean non-serializable values like NaN, inf, numpy types"""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    return data

@router.post("/analyze")
async def process_file(file: UploadFile = File(...)):
    updates = []

    try:
        updates.append("Uploaded file received")
        df = await file_to_dataframe(file)

        updates.append("Analyzer: Starting data profiling...")
        profile = await analyze(df)
        updates.append("Analyzer: Data profiling complete")

        updates.append("Operator: Starting data preprocessing...")
        operations = await operate(df, profile)
        updates.append("Operator: Data preprocessing complete")

        processed_df = operations.get("processed_df", df)

        updates.append("Scientist: Building and training model...")
        insights = await science(processed_df, profile)
        updates.append("Scientist: Model trained and evaluated")

        updates.append("Reporter: Generating PDF report...")
        report_path = await report(processed_df, profile, operations, insights)
        updates.append("Reporter: PDF report generated")

        # Sanitize entire payload before sending
        response = {
            "status": "success",
            "updates": updates,
            "profile": profile,
            "operations": operations,
            "insights": insights,
            "report_url": f"/download-report/{os.path.basename(report_path)}"
        }

        clean_response = sanitize_for_json(response)
        return JSONResponse(content=clean_response)

    except Exception as e:
        updates.append(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unhandled server error", "updates": updates}
        )

@router.get("/download-report/{filename}")
async def download_report(filename: str):
    reports_dir = Path("pdf_reports")
    file_path = reports_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")

async def file_to_dataframe(file: UploadFile) -> pd.DataFrame:
    tmp_path = None
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        if ext == '.csv':
            return pd.read_csv(tmp_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(tmp_path)
        elif ext == '.json':
            return pd.read_json(tmp_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
