from fastapi import APIRouter, UploadFile, File
from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science
from agents.reporter import report
import os
router = APIRouter()

@router.post("/analyze")
async def process_file(file: UploadFile = File(...)):
    df = await file_to_dataframe(file)
    
    # Parallel execution of agents
    profile = await analyze(df)
    operations = await operate(df)
    insights = await science(df)
    report_path = await report(df, insights)
    
    return {
        "profile": profile,
        "operations": operations,
        "insights": insights,
        "report_url": f"/download-report/{os.path.basename(report_path)}"
    }

async def file_to_dataframe(file: UploadFile):
    import pandas as pd
    import tempfile

    # Create a temporary file to store the uploaded file's contents
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Read the contents of the temporary file into a pandas DataFrame
    df = pd.read_csv(tmp_path)
    return df
