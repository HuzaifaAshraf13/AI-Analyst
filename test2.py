import pandas as pd
import asyncio
from typing import Dict
from agents.scientist import science

# Mock generate_insight to simulate Gemini API behavior
async def mock_generate_insight(prompt: str) -> str:
    # Simulate a mock response from Gemini API
    if "task" in prompt:
        return '''
{
    "task": "classification",
    "target_column": "target",
    "feature_columns": ["feature1", "feature2"],
    "rationale": "Based on the data characteristics."
}'''
    return '''```python
# [MODEL TRAINING]
# some random code
```'''

# Set up a basic test DataFrame
data = {
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'target': [0, 1, 0]
}
df = pd.DataFrame(data)

# Replace generate_insight in your original code with mock for testing
generate_insight = mock_generate_insight

# Run the test
async def test_science():
    analysis_results: Dict = {
        'ai_analysis': {
            'context': "Test data for classification task"
        }
    }

    result = await science(df, analysis_results)
    
    # Check if the result contains expected keys
    assert "model_type" in result
    assert "task" in result
    assert "metrics" in result
    assert "insights" in result
    assert "training_code" in result

    # Additional checks
    assert result["task"] == "classification"
    assert result["model_type"] in ["RandomForestClassifier", "RandomForestRegressor"]  # Based on your fallback

    print("Test passed!")

# Run the test
asyncio.run(test_science())
