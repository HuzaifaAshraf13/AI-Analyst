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
    updates = []

    try:
        updates.append("Uploaded file received")
        df = await file_to_dataframe(file)

        # Step 1: Analyzer
        updates.append("Analyzer: Starting data profiling...")
        profile = await analyze(df)
        updates.append("Analyzer: Data profiling complete")

        # Step 2: Operator
        updates.append("Operator: Starting data preprocessing...")
        operations = await operate(df, profile)  # Pass profile to operator
        updates.append("Operator: Data preprocessing complete")

        # Step 3: Scientist
        updates.append("Scientist: Building and training model...")
        insights = await science(df, profile)  # Pass profile to scientist
        updates.append("Scientist: Model trained and evaluated")

        # Step 4: Reporter
        updates.append("Reporter: Generating PDF report...")
        report_path = await report(df, profile, operations, insights)  # Pass all required arguments
        updates.append("Reporter: PDF report generated")

        return {
            "status": "success",
            "updates": updates,
            "profile": profile,
            "operations": operations,
            "insights": insights,
            "report_url": f"/download-report/{os.path.basename(report_path)}"
        }

    except Exception as e:
        updates.append(f"Error encountered: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "error": str(e),
            "updates": updates
        })

@router.get("/download-report/{filename}")
async def download_report(filename: str):
    """Endpoint to download generated PDF reports"""
    try:
        # Secure path construction for PDF reports
        reports_dir = Path("pdf_reports")
        file_path = reports_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def file_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """Convert uploaded file to pandas DataFrame"""
    tmp_path = None
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        if file_ext == '.csv':
            df = pd.read_csv(tmp_path)
        elif file_ext in ('.xls', '.xlsx'):
            df = pd.read_excel(tmp_path)
        elif file_ext == '.json':
            df = pd.read_json(tmp_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return df

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
