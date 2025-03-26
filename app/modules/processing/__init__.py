from .generic_processor import process_generic_file
from .pdf_processor import process_pdf
from .image_processor import process_image
from .audio_processor import process_audio
import mimetypes

def handle_file(filepath: str) -> list:
    """
    Recreate the original handle_file functionality
    
    Args:
        filepath: Path to file to process
        
    Returns:
        List of processed items
    """
    mime_type, _ = mimetypes.guess_type(filepath)
    items = []
    
    if not mime_type:
        items = process_generic_file(filepath)
    elif mime_type.startswith('application/pdf'):
        items = process_pdf(filepath)
    elif mime_type.startswith('audio'):
        items = process_audio(filepath)
    elif mime_type.startswith('image'):
        items = process_image(filepath)
    else:
        items = process_generic_file(filepath)
    
    return items