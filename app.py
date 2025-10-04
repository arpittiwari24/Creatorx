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

        # Extract captions with timestamps, split into 4-second chunks max
        captions = []
        MAX_DURATION = 5.0  # Maximum 4 seconds per caption

        for segment in segments:
            segment_duration = segment.end - segment.start

            if segment_duration <= MAX_DURATION:
                # Segment is already short enough
                captions.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            else:
                # Split long segments into smaller chunks
                words = segment.text.strip().split()
                if not words:
                    continue

                num_chunks = int(segment_duration / MAX_DURATION) + 1
                words_per_chunk = max(1, len(words) // num_chunks)

                current_start = segment.start
                chunk_duration = segment_duration / num_chunks

                for i in range(0, len(words), words_per_chunk):
                    chunk_words = words[i:i + words_per_chunk]
                    chunk_end = min(current_start + chunk_duration, segment.end)

                    captions.append({
                        "start": current_start,
                        "end": chunk_end,
                        "text": " ".join(chunk_words)
                    })

                    current_start = chunk_end

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
