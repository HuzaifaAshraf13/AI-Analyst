# agents/reporter.py
from utils.gemini_client import generate_insight
import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from datetime import datetime
from typing import Dict

REPORTS_DIR = "pdf_reports"  # Changed directory name
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

async def report(df: pd.DataFrame, insights: Dict) -> str:
    """Generate PDF report with visualizations and AI insights"""
    # Create PDF
    pdf = PDFReport()
    pdf.add_page()
    
    # Add metadata
    pdf.chapter_title(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add basic info
    pdf.chapter_title("Dataset Summary")
    pdf.chapter_body(f"Shape: {df.shape}\nColumns: {', '.join(df.columns)}")
    
    # Add visualizations
    plt.figure(figsize=(8, 6))
    df.hist()
    plot_path = os.path.join(REPORTS_DIR, "distributions.png")
    plt.savefig(plot_path)
    plt.close()
    
    pdf.chapter_title("Data Distributions")
    pdf.image(plot_path, x=10, w=190)
    os.remove(plot_path)  # Clean up temp image
    
    # Add AI insights
    pdf.chapter_title("AI Analysis Insights")
    pdf.chapter_body(insights.get('ai_insights', 'No insights generated'))
    
    # Save PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    pdf.output(report_path)
    
    return report_path