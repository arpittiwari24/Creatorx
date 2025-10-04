import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Whisper model once at startup
model = WhisperModel("base", compute_type="int8")

@app.post("/generate-captions")
async def generate_captions(video: UploadFile = File(...)):
    """
    Receives a video file and returns timed captions in JSON format
    """
    try:
        # Save uploaded video temporarily
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

        with open(video_path, "wb") as f:
            f.write(await video.read())

        # Transcribe video using Whisper
        segments, _ = model.transcribe(video_path, word_timestamps=True)

        # Extract captions with timestamps
        captions = []
        for segment in segments:
            captions.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # Clean up - remove uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

        return JSONResponse({
            "success": True,
            "captions": captions
        })

    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)

        raise HTTPException(status_code=500, detail=str(e))
