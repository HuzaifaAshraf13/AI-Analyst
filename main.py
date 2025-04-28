from fastapi import FastAPI,Request
from agents.routes import router as agent_router
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="AI Analyst Agents",
    description="Automated data analysis and reporting with AI agents",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")



# Include the agent router under the /api path
app.include_router(agent_router, prefix="/api")
# Route to render HTML
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})