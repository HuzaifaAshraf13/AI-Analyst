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

# === PDF Report Class (Improved) ===
class PDFReport(FPDF):
    def header(self):
        self.set_font('Times', 'B', 16)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 12, 'Comprehensive Data Analysis Report', 0, 1, 'C', fill=True)
        self.ln(5)
        self.set_draw_color(50, 50, 50)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def add_section_title(self, title: str):
        self.set_font('Times', 'B', 14)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(3)

    def chapter_body(self, body: str):
        self.set_font('Times', '', 12)
        cleaned_body = str(body).encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, cleaned_body)
        self.ln(4)

    def add_table(self, df: pd.DataFrame):
        max_cols_per_table = 6
        num_cols = len(df.columns)
        num_tables = (num_cols + max_cols_per_table - 1) // max_cols_per_table

        for t in range(num_tables):
            start_col = t * max_cols_per_table
            end_col = min((t + 1) * max_cols_per_table, num_cols)
            sub_df = df.iloc[:, start_col:end_col]

            self.set_font('Times', 'B', 8)  # Smaller font for headers
            col_width = (self.w - 2 * self.l_margin) / len(sub_df.columns)
            row_height = self.font_size * 1.6

            # Header
            self.set_fill_color(200, 220, 255)
            for col_name in sub_df.columns:
                readable_name = col_name.replace('_', ' ').title()
                self.cell(col_width, row_height, readable_name, border=1, align='C', fill=True)
            self.ln(row_height)

            # Rows
            self.set_font('Times', '', 8)  # Smaller font for data
            for _, row in sub_df.iterrows():
                for item in row:
                    self.cell(col_width, row_height, str(item), border=1, align='C')
                self.ln(row_height)

            self.ln(4)


    def add_bullets(self, points: list[str]):
        self.set_font('Times', '', 12)
        for point in points:
            self.cell(5)
            self.multi_cell(0, 8, f"- {point}")
        self.ln(4)

    def add_bold_bullet(self, title: str, detail: str):
        self.set_font('Times', 'B', 12)
        self.multi_cell(0, 8, f"{title}")
        self.set_font('Times', '', 12)
        self.multi_cell(0, 8, detail)
        self.ln(2)


# === Helper Functions ===
def format_output(data: Union[str, list, dict]) -> str:
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return "\n".join(map(str, data))
    elif isinstance(data, dict):
        return "\n".join([f"{k}: {v}" for k, v in data.items()])
    return str(data)

def save_feature_importance_plot(df: pd.DataFrame, feature_importance: Dict[str, float], path: str):
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    sns.barplot(x=importances, y=features, palette='crest')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_correlation_heatmap(df: pd.DataFrame, path: str):
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
    pdf = PDFReport()
    pdf.add_page()

    pdf.add_section_title(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pdf.add_section_title("Final Dataset Snapshot")
    pdf.chapter_body(f"Shape: {df.shape}\nColumns: {', '.join(df.columns)}")
    pdf.add_section_title("Sample Rows")
    pdf.add_table(df.sample(min(5, len(df))))

    try:
        hist_path = os.path.join(REPORTS_DIR, "histograms.png")
        plt.figure(figsize=(12, 10))
        df.select_dtypes(include='number').hist(bins=30, edgecolor='black', grid=False)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        pdf.add_section_title("Data Distributions")
        pdf.image(hist_path, x=15, w=180)
        os.remove(hist_path)

        heatmap_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
        save_correlation_heatmap(df, heatmap_path)
        pdf.add_section_title("Correlation Heatmap")
        pdf.image(heatmap_path, x=15, w=180)
        os.remove(heatmap_path)
    except Exception as e:
        pdf.chapter_body(f"Failed to generate visualizations: {e}")

    pdf.add_section_title("Analyzer Output")
    try:
        analyzer_out = format_output(profile)
        pdf.chapter_body(analyzer_out)
    except Exception as e:
        pdf.chapter_body(f"Analyzer Output Error: {e}")

    pdf.add_section_title("Operator Output")
    try:
        operator_out = format_output(operations)
        pdf.chapter_body(operator_out)
    except Exception as e:
        pdf.chapter_body(f"Operator Output Error: {e}")

    pdf.add_section_title("Scientist Output")
    try:
        scientist_info = insights
        if isinstance(scientist_info, dict):
            pdf.chapter_body("\n**Task:** " + scientist_info.get('task', '-'))
            pdf.chapter_body("**Target Column:** " + scientist_info.get('target_column', '-'))
            pdf.chapter_body("**Model Type:** " + scientist_info.get('model_type', '-'))

            pdf.chapter_body("**Metrics:**")
            for k, v in scientist_info.get('metrics', {}).items():
                pdf.chapter_body(f"- {k}: {v}")

            pdf.chapter_body("**Key Insights:**")
            for insight in scientist_info.get('insights', []):
                pdf.chapter_body(f"- {insight}")
        else:
            pdf.chapter_body(str(scientist_info))
    except Exception as e:
        pdf.chapter_body(f"Scientist Output Error: {e}")

    pdf.add_section_title("Comprehensive Research Summary")
    try:
        combined_prompt = f"""
Prepare a formal academic research report with sections: Abstract, Introduction, Data & Methodology, Exploratory Analysis, Modeling, Results, Discussion, Conclusion.

Data Sample:\n{df.head(5).to_string(index=False)}\n\nColumns Types:\n{df.dtypes.apply(str).to_dict()}\n\nAnalyzer Output:\n{analyzer_out}\n\nOperator Output:\n{operator_out}\n\nScientist Output:\n{scientist_info}\n\n
"""
        summary = generate_insight(combined_prompt)
        if isinstance(summary, list):
            summary = "\n".join(map(str, summary))
        pdf.chapter_body(summary)
    except Exception as e:
        pdf.chapter_body(f"Gemini AI summary failed: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, filename)
    pdf.output(report_path)

    return report_path
