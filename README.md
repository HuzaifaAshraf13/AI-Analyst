```markdown
# AI Data Analyst with Gemini

🚀 An AI-powered data analysis tool that automatically profiles datasets, suggests operations, generates insights, and creates reports using Google's Gemini API.

## Features

- **Smart Data Profiling**: Automatic detection of data types, missing values, and quality issues
- **Operation Suggestions**: AI recommends the most valuable pandas operations
- **Advanced Insights**: Identifies trends, patterns, and correlations
- **Visual Reporting**: Generates charts and comprehensive reports
- **Multi-format Support**: Works with CSV, Excel, and JSON files

## Tech Stack

- Python 3.9+
- FastAPI (Backend)
- Pandas (Data processing)
- Gemini API (AI analysis)
- Matplotlib (Visualizations)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-data-analyst.git
cd ai-data-analyst
```

2. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the FastAPI server:
```bash
uvicorn entry:app --reload
```

2. Access the API at `http://localhost:8000/docs` (Swagger UI)

3. Make a POST request to `/analyze` with your dataset file

### Example Request
```bash
curl -X POST -F "file=@your_dataset.csv" http://localhost:8000/analyze
```

### Example Response
```json
{
  "profile": {
    "basic_stats": {...},
    "ai_analysis": "The dataset contains..."
  },
  "operations": {
    "suggested_operations": "1. Handle missing values..."
  },
  "insights": {
    "statistical_summary": "...",
    "ai_insights": "Key trends identified..."
  },
  "report_url": "/download-report/analysis_report.txt"
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Upload dataset for analysis |
| `/download-report/{filename}` | GET | Download generated reports |

## Agent Architecture

The system uses four specialized AI agents:

1. **Analyzer**: Performs initial data profiling
2. **Operator**: Suggests data operations
3. **Scientist**: Generates advanced insights
4. **Reporter**: Creates visual reports

## Configuration

Edit `config.py` for:
- API timeout settings
- Report storage location
- Gemini model parameters

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
```

This README includes:

