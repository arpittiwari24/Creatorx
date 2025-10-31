import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

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

@app.get("/")
def test_function():
    return JSONResponse({
        "success" : "working"
    })

@app.post("/generate-captions")
async def generate_captions(
    video: UploadFile = File(...),
    maxLines: Optional[str] = Form("2"),
    maxWordsPerLine: Optional[str] = Form("0")
):
    """
    Receives a video file and returns timed captions in JSON format with formatting options

    Parameters:
    - video: The video file to transcribe
    - maxLines: Maximum number of lines per caption (default: 2)
    - maxWordsPerLine: Maximum words per line (default: 0 = unlimited)
    """
    try:
        # Parse formatting parameters
        max_lines = int(maxLines)
        max_words_per_line = int(maxWordsPerLine)

        # Calculate words per caption segment
        words_per_segment = max_lines * (max_words_per_line if max_words_per_line > 0 else 999)

        # Save uploaded video temporarily
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

        with open(video_path, "wb") as f:
            f.write(await video.read())

        # Transcribe video using Whisper
        segments, _ = model.transcribe(video_path, word_timestamps=True)

        # Extract captions with intelligent segmentation
        captions = []

        for segment in segments:
            # Extract word-level timestamps from the segment
            word_timestamps = []
            if hasattr(segment, 'words') and segment.words:
                word_timestamps = [
                    {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end
                    }
                    for word in segment.words
                ]

            # If no word timestamps available, fall back to simple text splitting
            if not word_timestamps:
                captions.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                })
                continue

            # Group words into caption chunks respecting word limits and natural boundaries
            current_chunk_words = []

            for i, word_data in enumerate(word_timestamps):
                # Check if adding this word would exceed the limit
                would_exceed_limit = len(current_chunk_words) >= words_per_segment

                # If we would exceed and we have words, create a caption first
                if would_exceed_limit and current_chunk_words:
                    chunk_text = " ".join([w["word"] for w in current_chunk_words])
                    chunk_start = current_chunk_words[0]["start"]
                    chunk_end = current_chunk_words[-1]["end"]

                    captions.append({
                        "start": chunk_start,
                        "end": chunk_end,
                        "text": chunk_text,
                        "words": current_chunk_words.copy()
                    })

                    current_chunk_words = []

                # Now add the current word
                current_chunk_words.append(word_data)

                # Check if this is the last word
                is_last_word = (i == len(word_timestamps) - 1)

                # Check for natural pause (gap between words > 0.3 seconds)
                has_pause = False
                if not is_last_word:
                    next_word = word_timestamps[i + 1]
                    gap = next_word["start"] - word_data["end"]
                    has_pause = gap > 0.3

                # Check for punctuation indicating natural break
                word_text = word_data["word"]
                has_punctuation = any(p in word_text for p in ['.', '!', '?', ',', ';'])

                # Determine when to break at natural boundaries
                should_break = False

                # Break at natural pauses if we have at least half the target words
                if has_pause and len(current_chunk_words) >= max(1, words_per_segment // 2):
                    should_break = True
                # Break at punctuation if we have at least half the target words
                elif has_punctuation and len(current_chunk_words) >= max(1, words_per_segment // 2):
                    should_break = True
                # Always break at the last word
                elif is_last_word:
                    should_break = True

                if should_break and current_chunk_words:
                    # Create caption from current chunk
                    chunk_text = " ".join([w["word"] for w in current_chunk_words])
                    chunk_start = current_chunk_words[0]["start"]
                    chunk_end = current_chunk_words[-1]["end"]

                    captions.append({
                        "start": chunk_start,
                        "end": chunk_end,
                        "text": chunk_text,
                        "words": current_chunk_words.copy()
                    })

                    current_chunk_words = []

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
