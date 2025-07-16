import os
import uuid
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
FRONTEND_DIR = "views"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load model once at startup
model = WhisperModel("base", compute_type="int8")  # Use "float16" if GPU available

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/upload/")
async def upload_video(file: UploadFile, font_size: int = Form(24), font_color: str = Form("white")):
    # Save uploaded video
    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{video_id}_captioned.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Transcribe using Faster-Whisper with word-level timestamps
    segments, _ = model.transcribe(input_path, word_timestamps=True)

    # Load original video
    clip = VideoFileClip(input_path)
    subtitles = []

    for segment in segments:
        # Text processing for 2-line display with word-level timing
        width, height = clip.size
        max_width = int(width * 0.8)  # Max width for each line
        
        # Get words with their timestamps
        words_with_timing = []
        if hasattr(segment, 'words') and segment.words:
            for word_info in segment.words:
                words_with_timing.append({
                    'word': word_info.word.strip(),
                    'start': word_info.start,
                    'end': word_info.end
                })
        else:
            # Fallback: split words evenly across segment duration
            words = segment.text.split()
            word_duration = (segment.end - segment.start) / len(words)
            for i, word in enumerate(words):
                words_with_timing.append({
                    'word': word,
                    'start': segment.start + (i * word_duration),
                    'end': segment.start + ((i + 1) * word_duration)
                })
        
        # Create chunks that fit in 2 lines with proper timing
        chunks = []
        current_chunk_words = []
        current_chunk_start = None
        
        for word_info in words_with_timing:
            test_chunk_words = current_chunk_words + [word_info]
            
            # Check if this chunk fits in 2 lines
            test_text = " ".join([w['word'] for w in test_chunk_words])
            test_words = test_text.split()
            
            # Try to fit in 2 lines
            lines = []
            current_line = []
            
            for test_word in test_words:
                test_line = current_line + [test_word]
                estimated_width = len(" ".join(test_line)) * font_size * 0.6
                
                if estimated_width > max_width and current_line:
                    lines.append(" ".join(current_line))
                    current_line = [test_word]
                else:
                    current_line = test_line
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # If it fits in 2 lines or less, add to current chunk
            if len(lines) <= 2:
                if current_chunk_start is None:
                    current_chunk_start = word_info['start']
                current_chunk_words = test_chunk_words
            else:
                # Save current chunk and start new one
                if current_chunk_words:
                    chunk_end = current_chunk_words[-1]['end']
                    chunks.append({
                        'words': current_chunk_words,
                        'start': current_chunk_start,
                        'end': chunk_end
                    })
                
                current_chunk_words = [word_info]
                current_chunk_start = word_info['start']
        
        # Add final chunk
        if current_chunk_words:
            chunk_end = current_chunk_words[-1]['end']
            chunks.append({
                'words': current_chunk_words,
                'start': current_chunk_start,
                'end': chunk_end
            })
        
        # Create text clips for each chunk with exact timing
        for chunk in chunks:
            chunk_text = " ".join([w['word'] for w in chunk['words']])
            
            # Split chunk into 2 lines
            chunk_words = chunk_text.split()
            lines = []
            current_line = []
            
            for word in chunk_words:
                test_line = current_line + [word]
                estimated_width = len(" ".join(test_line)) * font_size * 0.6
                
                if estimated_width > max_width and current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    current_line = test_line
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Join lines with newline (max 2 lines)
            final_text = "\n".join(lines[:2])
            
            txt_clip = TextClip(
                text=final_text, 
                font_size=font_size, 
                color=font_color, 
                size=(max_width, height // 5),  # Height for 2 lines
                method="caption",
                text_align="center",
                stroke_color="black",
                stroke_width=2
            )
            
            # Use exact word timing
            txt_clip = (
                txt_clip
                .with_start(chunk['start'])
                .with_end(chunk['end'])
                .with_position("center")
            )
            subtitles.append(txt_clip)

    # Create final video with subtitles (preserve audio)
    final_clip = CompositeVideoClip([clip, *subtitles])
    
    # Ensure audio is preserved from original clip
    final_clip = final_clip.with_audio(clip.audio)
    
    final_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Clean up
    clip.close()
    final_clip.close()

    return FileResponse(output_path, media_type="video/mp4", filename="captioned_video.mp4")