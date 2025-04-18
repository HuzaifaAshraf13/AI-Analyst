# agents/reporter.py
from utils.gemini_client import generate_insight
import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from datetime import datetime
from typing import Dict, Union

REPORTS_DIR = "pdf_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Data Analysis Report', 0, 1, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

async def report(df: pd.DataFrame, insights: Dict[str, Union[str, list]]) -> str:
    """Generate PDF report with visualizations and AI insights"""
    pdf = PDFReport()
    pdf.add_page()
    
    # Metadata
    pdf.chapter_title(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset summary
    pdf.chapter_title("Dataset Summary")
    pdf.chapter_body(f"Shape: {df.shape}\nColumns: {', '.join(df.columns)}")
    
    # Distributions
    try:
        plt.figure(figsize=(10, 8))
        df.hist(bins=30, edgecolor='black', grid=False)
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "distributions.png")
        plt.savefig(plot_path)
        plt.close()
        
        pdf.chapter_title("Data Distributions")
        pdf.image(plot_path, x=10, w=190)
        os.remove(plot_path)
    except Exception as e:
        pdf.chapter_body(f"Could not generate distribution plot: {e}")
    
    # AI Insights
    pdf.chapter_title("AI Analysis Insights")
    ai_text = insights.get('ai_insights', 'No insights generated')
    if isinstance(ai_text, list):
        ai_text = '\n'.join(ai_text)
    pdf.chapter_body(ai_text)
    
    # Save PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    pdf.output(report_path)
    
    return report_path
