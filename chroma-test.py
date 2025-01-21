from chromadb import Client
from chromadb.config import Settings

# Test directory
db_directory = "/Users/coreydavis/Research_Leave/get-warc/warcs/chromadb"
client = Client(Settings(
    persist_directory=db_directory,  # Directory for persistent storage
    anonymized_telemetry=False      # Optional: Disable telemetry
))

# Create a collection and add dummy data
collection = client.get_or_create_collection(name="test_collection")
collection.add(
    documents=["test document"],
    metadatas=[{"key": "value"}],
    ids=["test_id"]
)

print(f"Data added to ChromaDB collection. Check {db_directory} for persistence.")
