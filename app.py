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
    """Split text into chunks that fit in exactly 2 lines maximum"""
    words = text.split()
    if not words:
        return []
    
    # Estimate characters per line based on font size (more conservative)
    chars_per_line = max(20, 80 - font_size)  # Even more conservative for strict 2-line limit
    
    chunks = []
    line1_words = []
    line2_words = []
    line1_length = 0
    line2_length = 0
    
    i = 0
    while i < len(words):
        word = words[i]
        word_length = len(word)
        
        # Try to fit word in line 1
        space_needed = word_length + (1 if line1_length > 0 else 0)  # +1 for space
        
        if line1_length + space_needed <= chars_per_line:
            # Fits in line 1
            line1_words.append(word)
            line1_length += space_needed
        else:
            # Doesn't fit in line 1, try line 2
            space_needed_line2 = word_length + (1 if line2_length > 0 else 0)
            
            if line2_length + space_needed_line2 <= chars_per_line and len(line2_words) < 20:  # Limit line 2 to prevent overflow
                # Fits in line 2
                line2_words.append(word)
                line2_length += space_needed_line2
            else:
                # Doesn't fit in line 2 either, save current chunk and start new one
                if line1_words:  # Only save if we have content
                    chunk_text_parts = []
                    if line1_words:
                        chunk_text_parts.append(" ".join(line1_words))
                    if line2_words:
                        chunk_text_parts.append(" ".join(line2_words))
                    
                    chunks.append(" ".join(chunk_text_parts))
                
                # Start new chunk with current word
                line1_words = [word]
                line2_words = []
                line1_length = word_length
                line2_length = 0
        
        i += 1
    
    # Add final chunk if we have words left
    if line1_words:
        chunk_text_parts = []
        if line1_words:
            chunk_text_parts.append(" ".join(line1_words))
        if line2_words:
            chunk_text_parts.append(" ".join(line2_words))
        
        chunks.append(" ".join(chunk_text_parts))
    
    # Create caption objects with timing
    if not chunks:
        return []
    
    duration = end_time - start_time
    # Ensure minimum duration per chunk (0.8 seconds for readability)
    min_chunk_duration = 0.8
    chunk_duration = max(min_chunk_duration, duration / len(chunks))
    
    result = []
    current_time = start_time
    
    for i, chunk_text in enumerate(chunks):
        chunk_start = current_time
        chunk_end = min(current_time + chunk_duration, end_time)
        
        # Ensure we don't exceed the original end time
        if i == len(chunks) - 1:  # Last chunk
            chunk_end = end_time
        
        result.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text
        })
        
        current_time = chunk_end
    
    return result

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/thumbnail-creator")
async def thumbnail_creator():
    return FileResponse(os.path.join(FRONTEND_DIR, "thumbnail-creator.html"))

@app.get("/video-concatenate")
async def video_concatenate():
    return FileResponse(os.path.join(FRONTEND_DIR, "video-concatenate.html"))

@app.get("/aspect-ratio-converter")
async def aspect_ratio_converter():
    return FileResponse(os.path.join(FRONTEND_DIR, "aspect-ratio-converter.html"))

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
                size=(max_width , height // 5),  # Required for method="caption"
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
        
        # Process edited captions through 2-line splitting logic
        captions_data = []
        for caption in edit_request.captions:
            # Re-split each edited caption to ensure 2-line maximum
            if caption.text.strip():  # Only process non-empty captions
                split_chunks = split_text_into_chunks(
                    caption.text, 
                    caption.start, 
                    caption.end, 
                    edit_request.font_size
                )
                captions_data.extend(split_chunks)
            else:
                # Keep empty captions as-is (they'll be skipped in video generation)
                captions_data.append({
                    "start": caption.start,
                    "end": caption.end,
                    "text": caption.text
                })
        
        # Debug: Print processed caption data
        print(f"After 2-line processing: {len(captions_data)} caption chunks")
        for i, cap in enumerate(captions_data):
            print(f"Caption {i}: '{cap['text']}' ({cap['start']:.2f}-{cap['end']:.2f})")
        
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

def concatenate_videos_side_by_side(main_video_path: str, background_video_path: str, output_path: str):
    """Concatenate two videos side by side with 16:9 aspect ratio (optimized for speed)"""
    try:
        print(f"Loading videos: main={main_video_path}, bg={background_video_path}")
        main_clip = VideoFileClip(main_video_path)
        bg_clip = VideoFileClip(background_video_path)
        
        print(f"Main video: {main_clip.w}x{main_clip.h}, duration={main_clip.duration}s")
        print(f"Background video: {bg_clip.w}x{bg_clip.h}, duration={bg_clip.duration}s")
        
        # Target aspect ratio: 16:9 (width:height = 16/9)
        target_aspect_ratio = 16.0 / 9.0
        
        # Use a standard base resolution for consistent 16:9 ratio
        # Let's use gameplay video width but ensure perfect 16:9
        bg_width = bg_clip.w  # Keep gameplay video at full width
        
        # Calculate total width: bg_width represents 15% of total
        total_width = int(bg_width / 0.15)
        
        # Ensure total_width is divisible by 16 for perfect 16:9
        total_width = (total_width // 16) * 16
        
        # Calculate height for exact 16:9 ratio
        final_height = int(total_width * 9 / 16)
        
        # Calculate actual widths
        actual_bg_width = int(total_width * 0.15)  # 15% for gameplay
        actual_main_width = total_width - actual_bg_width  # 85% for main video
        
        print(f"Final dimensions: {total_width}x{final_height} (16:9)")
        print(f"Main video will be: {actual_main_width}x{final_height}")
        print(f"Background video will be: {actual_bg_width}x{final_height}")
        
        # Resize videos efficiently using MoviePy's resize method
        print("Resizing main video...")
        main_resized = main_clip.resized(new_size=(actual_main_width, final_height))
        
        print("Resizing background video...")
        bg_resized = bg_clip.resized(new_size=(actual_bg_width, final_height))
        
        # Loop background video if it's shorter than main video
        print("Adjusting video durations...")
        if bg_resized.duration < main_resized.duration:
            bg_resized = bg_resized.with_duration(main_resized.duration).loop()
        else:
            bg_resized = bg_resized.subclipped(0, main_resized.duration)
        
        # Position videos side by side
        print("Positioning videos...")
        main_positioned = main_resized.with_position((0, 0))
        bg_positioned = bg_resized.with_position((actual_main_width, 0))
        
        # Compose final video with 16:9 aspect ratio
        print("Compositing final video...")
        final_video = CompositeVideoClip([main_positioned, bg_positioned], size=(total_width, final_height))
        final_video = final_video.with_audio(main_clip.audio)  # Keep main video's audio
        
        # Write the output with speed optimizations
        print(f"Writing output to: {output_path}")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio-concat.m4a",
            remove_temp=True,
            # Speed optimizations
            preset="ultrafast",  # Fastest encoding preset
            ffmpeg_params=["-crf", "23"]  # Good quality/speed balance
        )
        
        print("Video concatenation completed successfully!")
        
        # Clean up
        main_clip.close()
        bg_clip.close()
        final_video.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error in concatenate_videos_side_by_side: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

@app.post("/concatenate-videos/")
async def concatenate_videos(
    main_video: UploadFile,
    background_video: UploadFile
):
    """Concatenate main video (85%) with background video (15%) side by side"""
    
    # Generate unique ID for this operation
    operation_id = str(uuid.uuid4())
    
    # Save uploaded videos
    main_input_path = os.path.join(UPLOAD_DIR, f"{operation_id}_main.mp4")
    bg_input_path = os.path.join(UPLOAD_DIR, f"{operation_id}_bg.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{operation_id}_concatenated.mp4")
    
    # Write main video
    with open(main_input_path, "wb") as f:
        f.write(await main_video.read())
    
    # Write background video
    with open(bg_input_path, "wb") as f:
        f.write(await background_video.read())
    
    try:
        print(f"Starting concatenation: main={main_input_path}, bg={bg_input_path}, output={output_path}")
        
        # Check if input files exist and have content
        if not os.path.exists(main_input_path) or os.path.getsize(main_input_path) == 0:
            raise Exception("Main video file is empty or missing")
        if not os.path.exists(bg_input_path) or os.path.getsize(bg_input_path) == 0:
            raise Exception("Background video file is empty or missing")
        
        # Perform video concatenation
        concatenate_videos_side_by_side(main_input_path, bg_input_path, output_path)
        
        # Check if output was created
        if not os.path.exists(output_path):
            raise Exception("Output video was not created")
        
        print(f"Concatenation successful: {output_path}")
        
        return JSONResponse({
            "success": True,
            "operation_id": operation_id,
            "video_url": f"/download/{operation_id}_concatenated.mp4"
        })
        
    except Exception as e:
        print(f"Error in concatenate_videos: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error concatenating videos: {str(e)}")
    
    finally:
        # Clean up input files
        try:
            if os.path.exists(main_input_path):
                os.remove(main_input_path)
            if os.path.exists(bg_input_path):
                os.remove(bg_input_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up files: {cleanup_error}")

def convert_video_aspect_ratio(input_path: str, output_path: str, target_ratio: str):
    """Convert video to specified aspect ratio"""
    try:
        print(f"Loading video for aspect ratio conversion: {input_path}")
        clip = VideoFileClip(input_path)
        
        print(f"Original video: {clip.w}x{clip.h}, duration={clip.duration}s")
        
        # Define common aspect ratios
        aspect_ratios = {
            "16:9": (16, 9),
            "9:16": (9, 16),  # Vertical/Portrait
            "4:3": (4, 3),
            "3:4": (3, 4),    # Vertical 4:3
            "1:1": (1, 1),    # Square
            "21:9": (21, 9),  # Ultra-wide
            "2:1": (2, 1),    # Cinema
            "16:10": (16, 10),
            "5:4": (5, 4)
        }
        
        if target_ratio not in aspect_ratios:
            raise ValueError(f"Unsupported aspect ratio: {target_ratio}")
        
        ratio_w, ratio_h = aspect_ratios[target_ratio]
        target_aspect = ratio_w / ratio_h
        
        # Get original dimensions
        original_w, original_h = clip.w, clip.h
        original_aspect = original_w / original_h
        
        print(f"Target aspect ratio: {target_ratio} ({target_aspect:.3f})")
        print(f"Original aspect ratio: {original_aspect:.3f}")
        
        # Determine how to fit the video into target aspect ratio
        if abs(original_aspect - target_aspect) < 0.001:
            # Already the correct aspect ratio
            print("Video already has target aspect ratio")
            converted_clip = clip
        elif original_aspect > target_aspect:
            # Original is wider, need to fit to target height and crop/pad width
            new_height = original_h
            new_width = int(new_height * target_aspect)
            
            # Make sure width is even for video encoding
            new_width = new_width + (new_width % 2)
            
            print(f"Cropping/padding to: {new_width}x{new_height}")
            
            if new_width < original_w:
                # Need to crop width (center crop)
                converted_clip = clip.resized(new_size=(new_width, new_height))
            else:
                # Need to pad width (add black bars on sides)
                converted_clip = clip.resized(new_size=(new_width, new_height))
        else:
            # Original is taller, need to fit to target width and crop/pad height
            new_width = original_w
            new_height = int(new_width / target_aspect)
            
            # Make sure height is even for video encoding
            new_height = new_height + (new_height % 2)
            
            print(f"Cropping/padding to: {new_width}x{new_height}")
            
            if new_height < original_h:
                # Need to crop height (center crop)
                converted_clip = clip.resized(new_size=(new_width, new_height))
            else:
                # Need to pad height (add black bars on top/bottom)
                converted_clip = clip.resized(new_size=(new_width, new_height))
        
        # Write the output
        print(f"Writing converted video to: {output_path}")
        converted_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio-ratio.m4a",
            remove_temp=True,
            preset="fast",
            ffmpeg_params=["-crf", "23"]
        )
        
        print("Aspect ratio conversion completed successfully!")
        
        # Clean up
        clip.close()
        converted_clip.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error in convert_video_aspect_ratio: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

@app.post("/convert-aspect-ratio/")
async def convert_aspect_ratio(
    video: UploadFile,
    aspect_ratio: str = Form(...)
):
    """Convert video to specified aspect ratio"""
    
    # Generate unique ID for this operation
    operation_id = str(uuid.uuid4())
    
    # Save uploaded video
    input_path = os.path.join(UPLOAD_DIR, f"{operation_id}_input.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{operation_id}_converted_{aspect_ratio.replace(':', '-')}.mp4")
    
    # Write video file
    with open(input_path, "wb") as f:
        f.write(await video.read())
    
    try:
        print(f"Starting aspect ratio conversion: input={input_path}, ratio={aspect_ratio}")
        
        # Check if input file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise Exception("Input video file is empty or missing")
        
        # Perform aspect ratio conversion
        convert_video_aspect_ratio(input_path, output_path, aspect_ratio)
        
        # Check if output was created
        if not os.path.exists(output_path):
            raise Exception("Converted video was not created")
        
        print(f"Aspect ratio conversion successful: {output_path}")
        
        return JSONResponse({
            "success": True,
            "operation_id": operation_id,
            "aspect_ratio": aspect_ratio,
            "video_url": f"/download/{os.path.basename(output_path)}"
        })
        
    except Exception as e:
        print(f"Error in convert_aspect_ratio: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error converting aspect ratio: {str(e)}")
    
    finally:
        # Clean up input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up input file: {cleanup_error}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed video files"""
    file_path = os.path.join(PROCESSED_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="video/mp4", filename=filename)