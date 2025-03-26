# test_uploads.py
import time
from backup.app import *

# Test files - replace with your actual paths
TEST_FILES = {
    "image": "3.jpg",
    "pdf": "paper.pdf",
    "docx": "Paper Results.docx",
    "mp3": "sample.mp3"
}

def test_file_processing():
    # Test individual file processors
    print("Testing PDF processing...")
    pdf_items = process_pdf(TEST_FILES["pdf"])
    assert len(pdf_items) > 0, "PDF processing failed"
    print(f"PDF processed {len(pdf_items)} items")
    
    print("\nTesting image processing...")
    img_items = process_image(TEST_FILES["image"])
    assert len(img_items) > 0, "Image processing failed"
    print(f"Image description: {img_items[0]['text'][:100]}...")

    print("\nTesting audio processing...")
    audio_items = process_audio(TEST_FILES["mp3"])
    assert len(audio_items) > 0, "Audio processing failed"
    print(f"Audio transcription chunks: {len(audio_items)}")

    print("\nTesting DOCX processing...")
    docx_items = process_generic_file(TEST_FILES["docx"])
    assert len(docx_items) > 0, "DOCX processing failed"
    print(f"DOCX text chunks: {len(docx_items)}")

def test_embeddings():
    print("\nTesting embedding generation...")
    generate_file_embeddings(list(TEST_FILES.values()))
    
    # Verify embeddings in Qdrant
    time.sleep(2)  # Allow time for upsert
    collections = qdrant_client.get_collections()
    assert any(c.name == "rag_embeddings" for c in collections.collections), "Collection missing"
    
    record_count = qdrant_client.count("rag_embeddings").count
    print(f"Stored embeddings: {record_count}")
    assert record_count > 0, "Embedding storage failed"

def test_cleanup():
    print("\nTesting cleanup...")
    cleanup_resources(perform_cleanup=True)
    
    # Verify image cleanup
    assert len(os.listdir("images")) == 0, "Image cleanup failed"
    
    # Verify Qdrant cleanup
    record_count = qdrant_client.count("rag_embeddings").count
    assert record_count == 0, "Qdrant cleanup failed"
    print("Cleanup successful")

if __name__ == "__main__":
    # Initialize fresh state
    initialize_vector_store()
    
    try:
        # test_file_processing()
        test_embeddings()
        test_cleanup()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")