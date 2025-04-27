from utils.gemini_client import generate_insight
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
from datetime import datetime
from typing import Dict, Union, Any

# === Setup Directories ===
REPORTS_DIR = "pdf_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# === PDF Report Class ===
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Comprehensive Data Analysis Report', 0, 1, 'C')

    def chapter_title(self, title: str):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body: str):
        self.set_font('Arial', '', 11)
        cleaned_body = str(body).encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, cleaned_body)
        self.ln()

# === Helper Functions ===
def format_output(data: Union[str, list, dict]) -> str:
    """Convert data to string format."""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return "\n".join(map(str, data))
    elif isinstance(data, dict):
        return "\n".join([f"{k}: {v}" for k, v in data.items()])
    return str(data)

def save_feature_importance_plot(df: pd.DataFrame, feature_importance: Dict[str, float], path: str):
    """Save feature importance plot to a file."""
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_correlation_heatmap(df: pd.DataFrame, path: str):
    """Save correlation heatmap to a file."""
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# === Main Reporting Function ===
async def report(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    operations: Dict[str, Any],
    insights: Dict[str, Any]
) -> str:
    """Generate a professional, academic-style PDF report based on transformed data."""
    
    # Clean insights
    clean_insights = dict(insights)
    if 'insights' in clean_insights and isinstance(clean_insights['insights'], list):
        clean_insights['insights'] = [
            i for i in clean_insights['insights']
            if 'fallback' not in str(i).lower()
        ]
    clean_insights.pop('warnings', None)

    # Create PDF
    pdf = PDFReport()
    pdf.add_page()

    # === 1. Metadata ===
    pdf.chapter_title(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === 2. Dataset Snapshot ===
    pdf.chapter_title("Final Dataset Snapshot")
    sample_text = df.sample(min(5, len(df))).to_string(index=False)
    pdf.chapter_body(
        f"Shape: {df.shape}\n"
        f"Columns: {', '.join(df.columns)}\n\n"
        f"Sample Rows:\n{sample_text}"
    )

    # === 3. Data Visualizations ===
    try:
        # Histograms
        hist_path = os.path.join(REPORTS_DIR, "histograms.png")
        plt.figure(figsize=(12, 10))
        df.select_dtypes(include='number').hist(bins=30, edgecolor='black', grid=False)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        pdf.chapter_title("Data Distributions")
        pdf.image(hist_path, x=10, w=190)
        os.remove(hist_path)

        # Correlation Heatmap
        heatmap_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
        save_correlation_heatmap(df, heatmap_path)
        pdf.chapter_title("Correlation Heatmap")
        pdf.image(heatmap_path, x=10, w=190)
        os.remove(heatmap_path)

    except Exception as e:
        pdf.chapter_body(f"Failed to generate visualizations: {e}")

    # === 4. Analyzer, Operator, Scientist Outputs ===
    pdf.chapter_title("Analyzer Output")
    try:
        analyzer_out = format_output(profile)
        pdf.chapter_body(f"The Analyzer agent performs an initial check on the dataset to ensure it is structurally valid and highlights any potential issues. Here are the results:\n\n{analyzer_out}")
    except Exception as e:
        pdf.chapter_body(f"Analyzer Output Error: {e}")

    pdf.chapter_title("Operator Output")
    try:
        operator_out = format_output(operations)
        pdf.chapter_body(f"The Operator agent performs preprocessing tasks on the dataset. This includes handling missing values, outliers, and other necessary transformations. Here are the operations performed:\n\n{operator_out}")
    except Exception as e:
        pdf.chapter_body(f"Operator Output Error: {e}")

    pdf.chapter_title("Scientist Output")
    try:
        scientist_out = format_output(clean_insights)
        pdf.chapter_body(f"The Scientist agent performs an in-depth analysis and modeling based on the cleaned data. It generates insights and suggests recommendations. Here are the findings and insights:\n\n{scientist_out}")
    except Exception as e:
        pdf.chapter_body(f"Scientist Output Error: {e}")

    # === 5. Final AI Research Paper-Style Summary ===
    pdf.chapter_title("Comprehensive Research Summary")
    try:
        combined_prompt = f"""
You are a professional Data Scientist preparing an academic research paper.

Use the following structure:

---
# Title: Comprehensive Data Analysis and Predictive Modeling of Electric Range

## Abstract
- Summarize objectives, dataset, methods, and findings.

## 1. Introduction
- Discuss the broader context and problem being solved.

## 2. Data and Methodology
- Explain dataset dimensions, types, preprocessing, and feature engineering.

## 3. Exploratory Data Analysis
- Highlight distributions, correlations, and initial insights.

## 4. Modeling Approach
- Describe ML models, tasks, validation strategies.

## 5. Results
- Report metrics (MSE, R²).
- Discuss feature importance.

## 6. Discussion
- Interpret results, limitations, future opportunities.

## 7. Conclusion
- Summarize key takeaways and recommendations.

---

# Data Snapshot:
{df.head(5).to_string(index=False)}

# Column Types:
{df.dtypes.apply(str).to_dict()}

# Analyzer Output:
{analyzer_out}

# Operator Output:
{operator_out}

# Scientist Output:
{scientist_out}

Compose a detailed academic report based on the provided information.
Use formal research-style English.
"""
        summary = await generate_insight(combined_prompt)
        if isinstance(summary, list):
            summary = "\n".join(map(str, summary))
        pdf.chapter_body(summary)
    except Exception as e:
        pdf.chapter_body(f"Gemini AI summary failed: {e}")

    # === 6. Save the Report ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, filename)
    pdf.output(report_path)

    return report_path
