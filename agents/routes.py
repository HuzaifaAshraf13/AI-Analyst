from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science
from agents.reporter import report
import os
import tempfile
import pandas as pd
from pathlib import Path

router = APIRouter()

@router.post("/analyze")
async def process_file(file: UploadFile = File(...)):
    """Endpoint to upload and analyze a data file"""
    try:
        df = await file_to_dataframe(file)
        
        # Process data through all AI agents
        profile = await analyze(df)
        operations = await operate(df)
        insights = await science(df)
        report_path = await report(df, insights)
        
        return {
            "status": "success",
            "profile": profile,
            "operations": operations,
            "insights": insights,
            "report_url": f"/download-report/{os.path.basename(report_path)}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/download-report/{filename}")
async def download_report(filename: str):
    """Endpoint to download generated reports"""
    try:
        # Secure path construction
        reports_dir = Path("reports")
        file_path = reports_dir / filename
        
        # Security checks
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def file_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """Convert uploaded file to pandas DataFrame"""
    try:
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Create temp file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Read based on file type
        if file_ext == '.csv':
            df = pd.read_csv(tmp_path)
        elif file_ext in ('.xls', '.xlsx'):
            df = pd.read_excel(tmp_path)
        elif file_ext == '.json':
            df = pd.read_json(tmp_path)
        else:
            raise ValueError("Unsupported file format")
            
        return df
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)