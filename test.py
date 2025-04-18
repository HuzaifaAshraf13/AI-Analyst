import pandas as pd
import asyncio
from agents.analyzer import analyze
from agents.operator import operate
from agents.scientist import science  # Importing scientist.py

async def test_analyzer_operator_scientist_pipeline():
    # Load sample CSV
    df = pd.read_csv("/home/eric/Desktop/Electric_Vehicle_Population_Data.csv")

    print("=== Step 1: Running Analyzer ===")
    analysis_results = await analyze(df)
    
    # Ensure analysis results are valid
    if not analysis_results:
        print("Error: Analyzer returned no valid results.")
        return
    
    print("Analysis Keys:", analysis_results.keys())
    print("Preprocessing Ready:", analysis_results.get("preprocessing_ready"))

    # Ensure preprocessing information is available
    preprocessing_ready = analysis_results.get("preprocessing_ready")
    if not preprocessing_ready:
        print("Error: No preprocessing steps recommended by analyzer.")
        return

    print("\n=== Step 2: Running Operator ===")
    operator_output = await operate(df, analysis_results=analysis_results)

    print("\nExecuted Operations:")
    for op in operator_output.get("executed_operations", []):
        print("-", op)

    print("\nSuggested Operations:")
    for sug in operator_output.get("suggested_operations", []):
        print("-", sug.get("purpose"))

    print("\nData Snapshot After Operations:")
    print(operator_output.get("data_snapshot"))

    # Pass the preprocessed DataFrame to scientist (this would be the next step)
    processed_df = operator_output.get("processed_df", df)  # Ensure this key exists

    if processed_df is None:
        print("Error: No processed data available from operator.")
        return

    print("\n=== Step 3: Running Scientist ===")
    # Run the scientist on the processed data
    scientist_results = await science(processed_df, analysis_results)
    
    # Ensure that the scientist results contain the expected information
    if not scientist_results:
        print("Error: Scientist returned no valid results.")
        return
    
    print("\nScientist Results:")
    print(scientist_results)

    # Check if model type and task are in the results
    model_type = scientist_results.get("model_type")
    task = scientist_results.get("task")
    
    if not model_type or not task:
        print("Error: Missing model type or task in scientist results.")
        return
    
    print(f"Model Type: {model_type}")
    print(f"Task: {task}")

    # Print the evaluation metrics and insights
    print("\nMetrics and Insights:")
    print(scientist_results.get("metrics"))
    print(scientist_results.get("insights"))

    print("\n=== Test Completed Successfully ===")
    print("\nProcessed DataFrame:")
    print(processed_df.head())

# Run the test
if __name__ == "__main__":
    asyncio.run(test_analyzer_operator_scientist_pipeline())
