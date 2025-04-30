import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Check environment variable
    environment = os.getenv("ENV", "local")  # Default is "local" if ENV not set

    if environment == "production":
        # Render.com or production environment
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port)
    else:
        # Local development
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
