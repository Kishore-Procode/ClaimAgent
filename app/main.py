import base64
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

import config
from models.model_manager import ModelManager
from pipeline.investigator import InvestigationPipeline
from models.schemas import ClaimRecord
from utils.csv_handler import load_claims_with_history

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

app = FastAPI(title="VisionClaim Investigator Dashboard")

# Ensure temporary directories exist
TEMP_DIR = Path("data/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize models and pipeline globally
logger.info("Initializing global model manager and pipeline...")
mm = ModelManager()
mm.load()

# Load history from sample claims for realistic history risk detection
history_map = {}
try:
    if Path(config.DEFAULT_SAMPLE_CSV).exists():
        _, history_map = load_claims_with_history(config.DEFAULT_SAMPLE_CSV, config.DEFAULT_DATA_DIR)
        logger.info(f"Loaded history for {len(history_map)} users from sample dataset.")
except Exception as e:
    logger.warning(f"Could not load sample history map: {e}")

pipeline = InvestigationPipeline(model_manager=mm, history_map=history_map)
pipeline.setup()

# Serve static files from app/static/
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def get_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Visual Investigator Dashboard</h1><p>index.html not found in static directory.</p>")

@app.get("/api/token-usage")
async def get_token_usage():
    """Return live session-level token usage from the global ModelManager.
    The frontend polls this every 5 s so the meter stays accurate across page reloads."""
    return JSONResponse(content=mm.get_session_usage_summary())


@app.websocket("/ws/investigate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    
    try:
        while True:
            # Expecting JSON frame with claim_text, claim_object, image_data, image_name
            data = await websocket.receive_json()
            
            claim_text = data.get("claim_text", "")
            claim_object = data.get("claim_object", "car")
            image_data_url = data.get("image_data", "")
            image_name = data.get("image_name", "upload.jpg")
            user_id = data.get("user_id", "web_user")
            
            logger.info(f"Received claim from {user_id} for {claim_object}")
            
            # 1. Process base64 image data url
            temp_image_path = None
            if image_data_url:
                try:
                    # Strip headers like data:image/jpeg;base64,
                    header, base64_data = image_data_url.split(",", 1)
                    # Extract extension from header (e.g. image/jpeg -> jpeg)
                    match = re.search(r"image/(\w+);", header)
                    ext = match.group(1) if match else "jpg"
                    
                    # Clean filename
                    safe_name = "".join(c for c in image_name if c.isalnum() or c in "._-").rstrip()
                    if not safe_name:
                        safe_name = f"upload.{ext}"
                    else:
                        # Ensure extension matches
                        if not safe_name.endswith(f".{ext}"):
                            safe_name = f"{Path(safe_name).stem}.{ext}"
                            
                    temp_image_path = TEMP_DIR / safe_name
                    
                    with open(temp_image_path, "wb") as f:
                        f.write(base64.b64decode(base64_data))
                        
                    logger.info(f"Saved uploaded image to: {temp_image_path}")
                except Exception as e:
                    logger.error(f"Failed to process uploaded image: {e}")
                    await websocket.send_json({"event": "error", "message": f"Image processing failed: {str(e)}"})
                    continue

            # 2. Build ClaimRecord
            img_paths_list = [str(temp_image_path)] if temp_image_path else []
            
            claim = ClaimRecord(
                user_id=user_id,
                image_paths=img_paths_list,
                user_claim=claim_text,
                claim_object=claim_object
            )

            # 3. Stream pipeline execution
            try:
                for stream_event in pipeline.investigate_stream(claim):
                    await websocket.send_json(stream_event)
            except Exception as e:
                logger.error(f"Pipeline stream failed: {e}", exc_info=True)
                await websocket.send_json({"event": "error", "message": f"Pipeline execution failed: {str(e)}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)

# Mount static files handler
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
