from fastapi import FastAPI
from agents.routes import router as agent_router

app = FastAPI(
    title="AI Analyst Agents",
    description="Automated data analysis and reporting with AI agents",
    version="1.0.0"
)

# Include the agent router under the /api path
app.include_router(agent_router, prefix="/api")
