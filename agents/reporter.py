from utils.gemini_client import generate_insight
import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from datetime import datetime
from typing import Dict, Union, Any

REPORTS_DIR = "pdf_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Data Analysis Report', 0, 1, 'C')
    
    def chapter_title(self, title: str):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)
    
    def chapter_body(self, body: str):
        self.set_font('Arial', '', 11)
        cleaned_body = str(body).encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, cleaned_body)
        self.ln()


def format_output(data: Union[str, list, dict]) -> str:
    """Format agent output into clean string."""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return "\n".join(map(str, data))
    elif isinstance(data, dict):
        return "\n".join([f"{k}: {v}" for k, v in data.items()])
    return str(data)


async def report(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    operations: Dict[str, Any],
    insights: Dict[str, Any]
) -> str:
    """Generate final PDF report based on transformed CSV data."""

    # === Clean out any fallback‑related messages from the scientist insights ===
    clean_insights = dict(insights)  # shallow copy
    # Remove any insight entries containing the word "fallback"
    if 'insights' in clean_insights and isinstance(clean_insights['insights'], list):
        clean_insights['insights'] = [
            i for i in clean_insights['insights']
            if 'fallback' not in str(i).lower()
        ]
    # Remove any warnings entirely (we don't want to show fallback warnings)
    clean_insights.pop('warnings', None)

    pdf = PDFReport()
    pdf.add_page()

    # === 1. Metadata ===
    pdf.chapter_title(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === 2. Final Dataset Info ===
    pdf.chapter_title("Final Dataset Snapshot")
    pdf.chapter_body(
        f"Shape: {df.shape}\n"
        f"Columns: {', '.join(df.columns)}\n\n"
        f"Sample:\n{df.head(5).to_string(index=False)}"
    )

    # === 3. Visualizations ===
    try:
        plt.figure(figsize=(12, 10))
        df.select_dtypes(include='number').hist(bins=30, edgecolor='black', grid=False)
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "distributions.png")
        plt.savefig(plot_path)
        plt.close()

        pdf.chapter_title("Data Distributions (Numeric Columns)")
        pdf.image(plot_path, x=10, w=190)
        os.remove(plot_path)
    except Exception as e:
        pdf.chapter_body(f"Failed to generate visualizations: {e}")

    # === 4. Agent Outputs ===
    analyzer_out = format_output(profile)
    operator_out = format_output(operations)
    scientist_out = format_output(clean_insights)

    pdf.chapter_title("Analyzer Output")
    pdf.chapter_body(analyzer_out)

    pdf.chapter_title("Operator Output")
    pdf.chapter_body(operator_out)

    pdf.chapter_title("Scientist Output")
    pdf.chapter_body(scientist_out)

    # === 5. Final AI Summary ===
    try:
        combined_prompt = f"""
You are an AI analyst. Based on the final dataset below and the AI pipeline results, write a structured analysis.

Final CSV Snapshot:
{df.head(5).to_string(index=False)}

Column Dtypes:
{df.dtypes.apply(str).to_dict()}

Pipeline Outputs:
[1] Analyzer Output:
{analyzer_out}

[2] Operator Output:
{operator_out}

[3] Scientist Output:
{scientist_out}

Now generate a final summary in the following structure:

---

### 🧠 Data Quality & Structure
- Describe column types, data quality issues, missing values, and outliers.
- Summarize the business context.

---

### 🧹 Preprocessing Actions
- List transformations that were executed or suggested.
- Mention any skipped operations needing review.

---

### 📊 Modeling & Results
- Summarize the ML task, target, model type, and performance metrics.
- Highlight key insights and feature importance.

---

### 📌 Key Recommendations
- Suggest next steps for improving the pipeline, model, or data quality.

---
"""
        summary = generate_insight(combined_prompt)
        if isinstance(summary, list):
            summary = "\n".join(map(str, summary))

        pdf.chapter_title("Final AI Summary")
        pdf.chapter_body(summary)

    except Exception as e:
        pdf.chapter_title("Final AI Summary")
        pdf.chapter_body(f"Gemini AI summary failed: {e}")

    # === 6. Save Report ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, filename)
    pdf.output(report_path)

    return report_path
