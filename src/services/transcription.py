import whisper_timestamped as whisper
import torch
import numpy as np
from typing import Dict, List

class TranscriptionService:
    def __init__(self):
        self.model = whisper.load_model("base.en", device="cuda" if torch.cuda.is_available() else "cpu")
    
    async def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Whisper model and return timestamped transcript
        """
        # Load and transcribe audio
        result = whisper.transcribe(
            self.model,
            audio_path,
            language="en",
            detect_speech_segments=True
        )
        
        # Process segments and identify speakers
        processed_segments = self._process_segments(result["segments"])
        
        return {
            "segments": processed_segments,
            "text": result["text"],
            "language": result["language"]
        }
    
    def _process_segments(self, segments: List) -> List[Dict]:
        """
        Process transcript segments and identify speakers
        """
        processed_segments = []
        for segment in segments:
            processed_segment = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "speaker": self._identify_speaker(segment),
                "words": segment.get("words", [])
            }
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _identify_speaker(self, segment: Dict) -> str:
        """
        Identify speaker (Customer/Salesperson) based on audio characteristics
        This is a placeholder for more sophisticated speaker diarization
        """
        # Implement speaker identification logic here
        # For now, returns placeholder
        return "Unknown"

transcription_service = TranscriptionService()

async def transcribe_audio(audio_path: str) -> Dict:
    """
    Wrapper function for transcription service
    """
    return await transcription_service.transcribe_audio(audio_path)