import os
import base64
from config import OLLAMA_BASE_URL
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import PIL.Image
from typing import List, Dict, Any


def describe_image(image_path: str) -> str:
    """
    Generates text description for an image using Ollama's Gemma
    
    Args:
        image_path: Path to the image file to describe
        
    Returns:
        Text description of the image
    """
    try:
        # Read and encode image
        with PIL.Image.open(image_path) as img:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create the message payload
        llm = ChatOllama(
            model="gemma3:4b",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            system="You are an expert at analyzing and describing images. Provide a detailed description including: "
                   "- Key objects and their spatial relationships\n"
                   "- Colors and visual style\n"
                   "- Text content (if any)\n"
                   "- Overall scene interpretation"
        )
        
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image in detail"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        print(f"Image description failed: {str(e)}")
        return "No description available"

def process_image(filepath: str) -> List[Dict[str, Any]]:
    """
    Processes user-uploaded images into standardized format
    
    Args:
        filepath: Path to the image file to process
        
    Returns:
        List containing single image description with metadata
    """
    try:
        os.makedirs("images", exist_ok=True)
        base_name = os.path.basename(filepath)
        image_name = f"images/uploaded_{base_name}"
        
        with open(filepath, 'rb') as src, open(image_name, 'wb') as dst:
            dst.write(src.read())
        
        return [{
            "type": "image",
            "text": describe_image(image_name),
            "metadata": {
                "source": image_name,
                "segment_type": "image"
            }
        }]
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return []