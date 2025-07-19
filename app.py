import os
import uuid
import json
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
FRONTEND_DIR = "views"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load model once at startup
model = WhisperModel("base", compute_type="int8")  # Use "float16" if GPU available

# Pydantic models for API
class Caption(BaseModel):
    start: float
    end: float
    text: str

class EditRequest(BaseModel):
    video_id: str
    captions: List[Caption]
    font_size: int = 24
    font_color: str = "white"
    text_height: int = 75

def split_text_into_chunks(text: str, start_time: float, end_time: float, font_size: int):
    """Split text into chunks that fit in maximum 2 lines"""
    words = text.split()
    if not words:
        return []
    
    # Estimate characters per line based on font size (rough calculation)
    chars_per_line = max(30, 120 - font_size)  # Smaller font = more chars per line
    max_chars_per_chunk = chars_per_line * 2  # 2 lines max
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed 2 lines
        test_length = current_length + len(word) + (1 if current_chunk else 0)  # +1 for space
        
        if test_length > max_chars_per_chunk and current_chunk:
            # Save current chunk and start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length = test_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Create caption objects with timing
    if not chunks:
        return []
    
    duration = end_time - start_time
    chunk_duration = duration / len(chunks)
    
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_start = start_time + (i * chunk_duration)
        chunk_end = start_time + ((i + 1) * chunk_duration)
        
        result.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text
        })
    
    return result

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/thumbnail-creator")
async def thumbnail_creator():
    return FileResponse(os.path.join(FRONTEND_DIR, "thumbnail-creator.html"))

@app.post("/upload/")
async def upload_video(file: UploadFile, font_size: int = Form(24), font_color: str = Form("white"), text_height: int = Form(75)):
    print(f"Received file: {file.filename}, font_size: {font_size}, font_color: {font_color}, text_height: {text_height}")
    # Save uploaded video
    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{video_id}_captioned.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Transcribe using Faster-Whisper with word-level timestamps
    segments, _ = model.transcribe(input_path, word_timestamps=True)
    
    # Extract captions for frontend editing - split into 2-line segments
    captions_data = []
    for segment in segments:
        # Split long segments into 2-line chunks
        segment_chunks = split_text_into_chunks(segment.text, segment.start, segment.end, font_size)
        captions_data.extend(segment_chunks)
    
    # Save captions data for later editing
    captions_path = os.path.join(PROCESSED_DIR, f"{video_id}_captions.json")
    with open(captions_path, "w") as f:
        json.dump(captions_data, f)

    # Generate video with captions
    processed_video = generate_captioned_video(input_path, captions_data, font_size, font_color, text_height)
    
    # Save processed video
    processed_video.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Clean up
    processed_video.close()

    return JSONResponse({
        "video_id": video_id,
        "captions": captions_data,
        "video_url": f"/download/{video_id}_captioned.mp4"
    })

def generate_captioned_video(input_path: str, captions_data: list, font_size: int, font_color: str, text_height: int):
    # Load original video
    clip = VideoFileClip(input_path)
    clip = clip.with_volume_scaled(3) # Increase volume by 50%
    subtitles = []
    
    # Get video dimensions
    width, height = clip.size

    for caption in captions_data:
        # Get caption text - use EXACTLY what user provided
        caption_text = caption.get('text', '')
        
        # Skip only if completely empty
        if not caption_text:
            continue
        
        # Create text clip with user's exact text
        try:
            max_width = int(width * 0.8)  # 80% of video width
            txt_clip = TextClip(
                text=caption_text, 
                font_size=font_size, 
                color=font_color, 
                size=(max_width, None),  # Required for method="caption"
                method="caption",
                text_align="center",
                stroke_color="black",
                stroke_width=2
            )
            
            # Position and timing
            constrained_height = max(50, min(90, text_height))
            y_position = height * (constrained_height / 100)
            
            txt_clip = (
                txt_clip
                .with_start(caption['start'])
                .with_end(caption['end'])
                .with_position(("center", y_position))
            )
            subtitles.append(txt_clip)
            
        except Exception as e:
            print(f"Error creating caption: {e}")
            continue

    # Create final video with subtitles (preserve audio)
    final_clip = CompositeVideoClip([clip, *subtitles])
    final_clip = final_clip.with_audio(clip.audio)
    
    return final_clip

@app.post("/edit-captions/")
async def edit_captions(edit_request: EditRequest):
    """Re-process video with edited captions"""
    try:
        print(f"Edit request received for video_id: {edit_request.video_id}")
        print(f"Number of captions: {len(edit_request.captions)}")
        print(f"Font settings: size={edit_request.font_size}, color={edit_request.font_color}, height={edit_request.text_height}")
        
        # Get original video path
        input_path = os.path.join(UPLOAD_DIR, f"{edit_request.video_id}.mp4")
        output_path = os.path.join(PROCESSED_DIR, f"{edit_request.video_id}_edited.mp4")
        
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="Original video not found")
        
        # Convert captions to dict format
        captions_data = [
            {
                "start": caption.start,
                "end": caption.end,
                "text": caption.text
            }
            for caption in edit_request.captions
        ]
        
        # Debug: Print caption data
        for i, cap in enumerate(captions_data):
            print(f"Caption {i}: '{cap['text']}' ({cap['start']}-{cap['end']})")
        
        # Generate new video with edited captions
        processed_video = generate_captioned_video(
            input_path, 
            captions_data, 
            edit_request.font_size, 
            edit_request.font_color, 
            edit_request.text_height
        )
        
        # Save processed video
        processed_video.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )
        
        # Clean up
        processed_video.close()
        
        return JSONResponse({
            "success": True,
            "video_url": f"/download/{edit_request.video_id}_edited.mp4"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed video files"""
    file_path = os.path.join(PROCESSED_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="video/mp4", filename=filename)