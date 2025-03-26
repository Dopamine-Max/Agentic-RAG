import whisper
from typing import List, Dict, Any
from modules.processing.generic_processor import text_splitter

transcribe_model = whisper.load_model("base")

def process_audio(filepath: str) -> List[Dict[str, Any]]:
    """
    Transcribes and processes audio files into text chunks with metadata.
    
    Args:
        filepath: Path to the audio file to process
        
    Returns:
        List of dictionaries containing audio transcript chunks and metadata
    """
    try:
        result = transcribe_model.transcribe(filepath)
        return [{
            "type": "audio",
            "text": chunk,
            "metadata": {
                "source": filepath,
                "segment_type": "audio"
            }
        } for chunk in text_splitter.split_text(result["text"])]
    except Exception as e:
        print(f"Audio processing failed: {str(e)}")
        return []