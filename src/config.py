# It Creates a configuration module.

# It Creates a configuration module.

import os
import getpass
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()

# API Keys with `getpass` fallback (only for OpenAI)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass.getpass("🔑 Enter OpenAI API Key: ")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Validate API keys (Only raise error for Pinecone if missing)
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Set PINECONE_API_KEY in your environment variables.")
if not OPENAI_API_KEY:
    print("⚠️ OpenAI API key is missing. You will need to enter it manually when prompted.")

# Global Embeddings Model (Used by all tools)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              openai_api_key=OPENAI_API_KEY)

# Pinecone Configuration
INDEX_NAME = "research-knowledge"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"🛠 Creating Pinecone index: {INDEX_NAME}...")
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(INDEX_NAME, dimension=1536, metric="cosine", spec=spec)
    print("✅ Pinecone index created.")
else:
    print(f"✅ Pinecone index '{INDEX_NAME}' already exists.")

