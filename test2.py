import pandas as pd
import asyncio
from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science
from agents.reporter import report
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_full_pipeline():
    try:
        # Load sample CSV
        df = pd.read_csv("/home/eric/Desktop/Electric_Vehicle_Population_Data.csv")

        print("=== Step 1: Running Analyzer ===")
        analysis_results = await analyze(df)
        
        if not analysis_results:
            print("Error: Analyzer returned no valid results.")
            return
        
        print("Analysis Keys:", analysis_results.keys())
        print("Preprocessing Ready:", analysis_results.get("preprocessing_ready"))

        print("\n=== Step 2: Running Operator ===")
        operator_output = await operate(df, analysis_results=analysis_results)

        if not operator_output:
            print("Error: Operator returned no valid results.")
            return

        print("\nExecuted Operations:")
        for op in operator_output.get("executed_operations", []):
            print("-", op)

        print("\nSuggested Operations:")
        for sug in operator_output.get("suggested_operations", []):
            print("-", sug.get("purpose"))

        print("\nData Snapshot After Operations:")
        print(operator_output.get("data_snapshot"))

        processed_df = operator_output.get("processed_df", df)
        if processed_df is None:
            print("Error: No processed data available from operator.")
            return

        print("\n=== Step 3: Running Scientist ===")
        scientist_results = await science(processed_df, analysis_results)
        
        if not scientist_results:
            print("Error: Scientist returned no valid results.")
            return

        print("\nScientist Results:")
        print(scientist_results)

        print(f"Model Type: {scientist_results.get('model_type')}")
        print(f"Task: {scientist_results.get('task')}")
        print("\nMetrics and Insights:")
        print(scientist_results.get("metrics"))
        print(scientist_results.get("insights"))

        print("\n=== Step 4: Running Reporter ===")
        
        # Ensure the pdf_reports directory exists
        report_dir = 'pdf_reports'
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)  # Create the directory if it doesn't exist

        # Generate and save the report
        report_path = await report(
            df=processed_df,
            profile=analysis_results,
            operations=operator_output,
            insights=scientist_results
        )

        if report_path:
            print(f"\n✅ PDF Report Generated: {report_path}")
        else:
            print("\n⚠️ Report generation failed or returned no path.")
        
        print("\n=== Full Test Completed Successfully ===")
    
    except Exception as e:
        print(f"❌ Test failed due to error: {e}")
        logger.error(f"Error during the full pipeline execution: {e}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
