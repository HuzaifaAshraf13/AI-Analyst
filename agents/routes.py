from fastapi import APIRouter, UploadFile, File, HTTPException,Request
from fastapi.responses import FileResponse, JSONResponse,StreamingResponse
import pandas as pd
import tempfile
import os
from pathlib import Path
import numpy as np
import asyncio

from typing import List

from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science
from agents.reporter import report

router = APIRouter()


progress_messages: List[asyncio.Queue] = []
def send_progress(message: str):
    """Broadcast progress to all SSE clients."""
    for queue in progress_messages:
        queue.put_nowait(message)

def sanitize_for_json(data):
    if isinstance(data, pd.DataFrame):
        return [sanitize_for_json(row) for row in data.to_dict(orient="records")]
    elif isinstance(data, pd.Series):
        return sanitize_for_json(data.to_dict())
    elif isinstance(data, dict):
        return {str(k): sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (np.integer, np.floating)):
        val = data.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    return data

@router.get("/progress")
async def progress_stream(request: Request):
    queue = asyncio.Queue()
    progress_messages.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                message = await queue.get()
                yield f"data: {message}\n\n"
        finally:
            progress_messages.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/analyze")
async def process_file(file: UploadFile = File(...)):
    try:
        send_progress("Fiona is receiving your file...")
        df = await file_to_dataframe(file)

        send_progress("Fiona is profiling your data...")
        profile = await analyze(df)

        send_progress("Fiona is preprocessing the data...")
        operations = await operate(df, profile)
        processed_df = operations.get("processed_df", df)

        send_progress("Fiona is ruuning model...")
        insights = await science(processed_df, profile)

        send_progress("Fiona is generating the report...")
        report_path = await report(processed_df, profile, operations, insights)

        send_progress("Fiona is done! Redirecting...")

        response = {
            "status": "success",
            "profile": profile,
            "operations": operations,
            "insights": insights,
            "report_url": f"/api/download-report/{os.path.basename(report_path)}"
        }

        return JSONResponse(content=sanitize_for_json(response))

    except Exception as e:
        send_progress("Something went wrong 😢")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/download-report/{filename}")
async def download_report(filename: str):
    reports_dir = Path("pdf_reports")
    file_path = reports_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(path=file_path, filename=filename, media_type="application/pdf",headers={"Content-Disposition": f'inline; filename="{filename}"'}
)

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
