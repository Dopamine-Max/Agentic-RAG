import pymupdf
import os
from typing import List, Dict, Any
from modules.processing.image_processor import describe_image
from modules.processing.generic_processor import process_text
    
# Save images
def process_pdf_images(doc: pymupdf.Document, page: pymupdf.Page, page_num: int) -> List[Dict[str, Any]]:
    """
    Extracts and processes images from a PDF page
    
    Args:
        doc: PyMuPDF Document object
        page: PyMuPDF Page object
        page_num: Zero-based page number
        
    Returns:
        List of image descriptions with metadata
    """
    
    os.makedirs("images", exist_ok=True)
    items = []
    
    for img_idx, img in enumerate(page.get_images()):
        try:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # Save PDF-extracted image
            image_name = f"images/pdf_{os.path.basename(doc.name)}_p{page_num}_i{img_idx}.png"
            pix.save(image_name)
            
            # Get description from saved image file
            description = describe_image(image_name)
            
            items.append({
                "type": "image",
                "text": description,
                "metadata": {
                    "source": doc.name,
                    "page": page_num + 1,
                    "segment_type": "image",
                    "image_path": image_name
                }
            })
        except Exception as e:
            print(f"PDF image processing error: {str(e)}")
    
    return items

def process_pdf(filepath: str) -> List[Dict[str, Any]]:
    """
    Processes PDF documents into structured content chunks
    
    Args:
        filepath: Path to the PDF file
        
    Returns:
        Combined list of text, and image content with metadata
    """
    
    doc = pymupdf.open(filepath)
    num_pages=len(doc)
    items = []
    
    for page_num in range(num_pages):
        
        page = doc[page_num]
        page_items = []
        
        # Process text
        text = page.get_text()
        page_items.extend(process_text(text, page_num, filepath))
        
        # Process images
        page_items.extend(process_pdf_images(doc, page, page_num))
        
        items.extend(page_items)
    
    return items