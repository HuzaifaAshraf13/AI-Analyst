# agents/reporter.py
from utils.gemini_client import generate_insight
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

async def report(df: pd.DataFrame, insights: Dict) -> str:
    """Generate visual report with AI commentary"""
    # Generate basic visualizations
    plt.figure(figsize=(10, 6))
    df.hist()
    plot_path = os.path.join(REPORTS_DIR, "distributions.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Get AI commentary
    prompt = f"""Create a comprehensive data report including:
    1. Executive summary
    2. Key findings
    3. Visualization interpretation
    4. Recommendations
    
    Insights to include:
    {insights}
    """
    
    try:
        report_content = generate_insight(prompt)
        
        # Save report to file
        report_path = os.path.join(REPORTS_DIR, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_content)
            f.write(f"\n\n[Visualization]\n{plot_path}")
        
        return report_path
    except Exception as e:
        raise Exception(f"Report generation failed: {str(e)}")