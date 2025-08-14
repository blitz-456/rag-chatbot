import os
import hashlib
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

DATA_STORAGE = os.getenv("DATA_STORAGE")
USE_QDRANT_CLOUD = os.getenv("USE_QDRANT_CLOUD", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "init-test-docs")

print(f"Data storage path: {DATA_STORAGE}")
print(f"Use Qdrant Cloud? {USE_QDRANT_CLOUD}")

# -------------------------
# Helper Functions
# -------------------------
def generate_file_hash(file_path: str) -> str:
    """Generate a unique hash for a file based on its content."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return None

def get_processed_files(client, collection_name):
    """Fetch all file hashes that have already been processed."""
    processed_files = set()
    offset = None
    
    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                limit=1000,
                scroll_filter=None,
                offset=offset
            )
            for p in points:
                if "file_hash" in p.payload:
                    processed_files.add(p.payload["file_hash"])
            if offset is None:
                break
    except Exception as e:
        print(f"Error fetching processed files: {e}")
        # If collection doesn't exist or error occurs, return empty set
        return set()
    
    return processed_files

def get_files_to_process(data_path):
    """Get list of files to process with their hashes."""
    files_info = []
    
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return files_info
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(data_path, filename)
            file_hash = generate_file_hash(file_path)
            if file_hash:
                files_info.append({
                    'filename': filename,
                    'file_path': file_path,
                    'file_hash': file_hash
                })
    
    return files_info

# -------------------------
# Get files to process
# -------------------------
all_files = get_files_to_process(DATA_STORAGE)
print(f"Found {len(all_files)} PDF files in directory:")
for file_info in all_files:
    print(f"  - {file_info['filename']} (hash: {file_info['file_hash'][:8]}...)")

if not all_files:
    print("❌ No PDF files found to process.")
    exit()

# -------------------------
# Embedding model
# -------------------------
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# -------------------------
# Qdrant connection
# -------------------------
if USE_QDRANT_CLOUD:
    print(f"Connecting to Qdrant Cloud at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("Using local Qdrant (embedded or via local server).")
    local_path = os.getenv("QDRANT_LOCAL_PATH")
    client = QdrantClient(path=local_path)

# -------------------------
# Create collection if missing
# -------------------------
vector_size = len(embed_model.get_text_embedding("test"))

try:
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION_NAME for c in collections)
except:
    collection_exists = False

if not collection_exists:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created.")
else:
    print(f"✅ Collection '{COLLECTION_NAME}' already exists.")

# -------------------------
# Check for already processed files
# -------------------------
processed_file_hashes = get_processed_files(client, COLLECTION_NAME)
print(f"Found {len(processed_file_hashes)} previously processed files in Qdrant.")

# Filter out already processed files
new_files_to_process = []
for file_info in all_files:
    if file_info['file_hash'] not in processed_file_hashes:
        new_files_to_process.append(file_info)
    else:
        print(f"⚠️ Skipping already processed file: {file_info['filename']}")

if not new_files_to_process:
    print("⚠️ No new files to process. All files have already been indexed.")
    exit()

print(f"Processing {len(new_files_to_process)} new files...")

# -------------------------
# Load and process only new files
# -------------------------
file_extractor = {".pdf": PyMuPDFReader()}

# Process each new file separately to add file metadata
all_new_documents = []
for file_info in new_files_to_process:
    print(f"Loading file: {file_info['filename']}")
    
    # Load documents from this specific file
    documents = SimpleDirectoryReader(
        input_files=[file_info['file_path']], 
        file_extractor=file_extractor
    ).load_data()
    
    # Add file metadata to each document chunk
    for doc in documents:
        doc.metadata.update({
            "file_hash": file_info['file_hash'],
            "source_file": file_info['filename'],
            "file_path": file_info['file_path']
        })
    
    all_new_documents.extend(documents)
    print(f"  → Generated {len(documents)} chunks from {file_info['filename']}")

# -------------------------
# Index new documents
# -------------------------
if all_new_documents:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embed_model
    )
    storage = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        all_new_documents,
        storage_context=storage,
        embed_model=embed_model
    )
    
    print(f"✅ Successfully indexed {len(new_files_to_process)} new files:")
    for file_info in new_files_to_process:
        print(f"  - {file_info['filename']}")
    print(f"✅ Total chunks added: {len(all_new_documents)}")
else:
    print("⚠️ No new documents to add.")

print("🎉 Indexing complete!")