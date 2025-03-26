from qdrant_client import QdrantClient, models
from config import QDRANT_URL, QDRANT_API_KEY

#Intialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def initialize_vector_store():
    """
    Initializes or resets the Qdrant collection for storing document embeddings.
    
    Creates a new collection 'rag_embeddings' with configuration matching
    the nomic-embed-text model dimensions.
    """
    
    try:
        
        if not qdrant_client.get_collections():
            raise ConnectionError("Failed to connect to Qdrant cluster")
        
        if not qdrant_client.collection_exists("rag_embeddings"):
            qdrant_client.create_collection(
                collection_name="rag_embeddings",
                vectors_config=models.VectorParams(
                    size=768,  # Dimension for nomic-embed-text
                    distance=models.Distance.COSINE
                )
            )
            print("Vector store initialized successfully")
        
    except Exception as e:
        print(f"Vector store initialization failed: {str(e)}")
        raise