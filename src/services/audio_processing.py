import os
from pathlib import Path
from pydub import AudioSegment
from fastapi import UploadFile, HTTPException
import uuid

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create uploads directory if it doesn't exist
UPLOAD_DIR.mkdir(exist_ok=True)

async def process_audio_file(file: UploadFile) -> str:
    """
    Process uploaded audio file:
    1. Validate file
    2. Save to disk
    3. Convert to WAV if necessary
    4. Return path to processed file
    """
    # Validate file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Get file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Convert to WAV if necessary
    if ext != ".wav":
        wav_path = file_path.with_suffix(".wav")
        try:
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_path, format="wav")
            os.remove(file_path)  # Remove original file
            file_path = wav_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting file: {str(e)}")
    
    return str(file_path)