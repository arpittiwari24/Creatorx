import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

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

def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds using ffprobe
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        raise Exception(f"Failed to get video duration: {str(e)}")


def split_into_lines(words_data: List[dict], max_words_per_line: int) -> List[List[dict]]:
    """
    Split words into lines based on max_words_per_line constraint
    Returns list of lines, where each line is a list of word dictionaries
    """
    if max_words_per_line <= 0:
        # No limit - return all words as single line
        return [words_data]

    lines = []
    current_line = []

    for word_data in words_data:
        current_line.append(word_data)

        # Check if we've reached the word limit for this line
        if len(current_line) >= max_words_per_line:
            lines.append(current_line)
            current_line = []

    # Add remaining words as final line
    if current_line:
        lines.append(current_line)

    return lines


def create_caption_from_lines(lines: List[List[dict]]) -> dict:
    """
    Create a caption object from a list of lines
    """
    if not lines:
        return None

    all_words = [word for line in lines for word in line]

    # Format text with line breaks
    text_lines = []
    for line in lines:
        line_text = " ".join([w["word"] for w in line])
        text_lines.append(line_text)

    caption_text = "\n".join(text_lines)

    return {
        "start": all_words[0]["start"],
        "end": all_words[-1]["end"],
        "text": caption_text,
        "words": all_words
    }


@app.get("/")
def test_function():
    return JSONResponse({
        "success": "working"
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
    video_path = None
    try:
        # Parse formatting parameters
        max_lines = int(maxLines)
        max_words_per_line = int(maxWordsPerLine)

        print(f"Processing with maxLines={max_lines}, maxWordsPerLine={max_words_per_line}")

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

            # If no word timestamps available, fall back to simple text
            if not word_timestamps:
                captions.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                })
                continue

            # Process words into captions respecting maxLines and maxWordsPerLine
            current_caption_words = []

            for i, word_data in enumerate(word_timestamps):
                current_caption_words.append(word_data)

                # Check if this is the last word in the segment
                is_last_word = (i == len(word_timestamps) - 1)

                # Split current words into lines
                lines = split_into_lines(current_caption_words, max_words_per_line)

                # Check if we've exceeded max_lines
                exceeded_max_lines = max_lines > 0 and len(lines) > max_lines

                # Check for natural pause (gap between words > 0.3 seconds)
                has_pause = False
                if not is_last_word:
                    next_word = word_timestamps[i + 1]
                    gap = next_word["start"] - word_data["end"]
                    has_pause = gap > 0.3

                # Check for punctuation indicating natural break
                word_text = word_data["word"]
                has_punctuation = any(p in word_text for p in ['.', '!', '?'])

                # Decide when to create a caption
                should_create_caption = False

                if exceeded_max_lines:
                    # Remove the last word and create caption with remaining words
                    current_caption_words.pop()
                    should_create_caption = True
                elif is_last_word:
                    # Always create caption at the end
                    should_create_caption = True
                elif has_punctuation and len(current_caption_words) >= 3:
                    # Create caption at sentence end if we have enough words
                    should_create_caption = True
                elif has_pause and len(current_caption_words) >= 3:
                    # Create caption at natural pause if we have enough words
                    should_create_caption = True

                if should_create_caption and current_caption_words:
                    # Split words into lines and create caption
                    lines = split_into_lines(current_caption_words, max_words_per_line)

                    # If we have maxLines limit, only take first maxLines
                    if max_lines > 0 and len(lines) > max_lines:
                        lines = lines[:max_lines]

                    caption = create_caption_from_lines(lines)
                    if caption:
                        captions.append(caption)

                    # Reset for next caption, but keep the word we removed if we exceeded
                    if exceeded_max_lines:
                        current_caption_words = [word_data]
                    else:
                        current_caption_words = []

        # Clean up - remove uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

        print(f"Generated {len(captions)} captions for video {video.filename}")

        return JSONResponse({
            "success": True,
            "captions": captions
        })

    except Exception as e:
        # Clean up on error
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
