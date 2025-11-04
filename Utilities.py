# Utils for handling operations

import os
import hashlib
import re
import json
import requests
import ollama
import chromadb
from datetime import datetime as pydatetime, timezone, timedelta
from typing import List, Optional, Dict

# Defaults for Chroma / Ollama; can be overridden by environment variables
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "localrag")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "phi")  # Use phi for embeddings as it's smaller
# Keep a conservative default LLM model name to avoid implicitly forcing a large model
# that may require excessive memory. Prefer setting LLM_MODEL via environment variable
# when you need a specific model. If you want to provide a list of fallback models
# (tried in order when memory errors occur), set the `LLM_FALLBACKS` env var to a
# comma-separated list.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma:2b")

###
## Document Splitting Functions
###

def split_text_on_newlines(text: str) -> List[str]:
    """Split text into paragraphs based on newlines."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def split_text_on_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Basic sentence splitting - could be improved with nltk or spacy
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def fixed_size_chunking(document: str, chunk_size: int) -> List[str]:
    """Split document into fixed-size chunks.
    
    Args:
        document: Text to split
        chunk_size: Maximum number of characters per chunk
    """
    chunks = []
    for i in range(0, len(document), chunk_size):
        chunks.append(document[i:i + chunk_size])
    return chunks

def sliding_chunking(document: str, chunk_size: int, overlap: int) -> List[str]:
    """Split document into overlapping chunks.
    
    Args:
        document: Text to split
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return chunks

def recursive_chunking(document: str, max_chunk_size: int, separators: Optional[List[str]] = None) -> List[str]:
    """Split document recursively using a hierarchy of separators.
    
    Args:
        document: Text to split
        max_chunk_size: Maximum number of characters per chunk
        separators: List of strings to use as separators, in priority order.
                   Defaults to ['\n\n', '.', ',']
    """
    if separators is None:
        separators = ['\n\n', '.', ',']
    
    def split_recursively(text: str, sep_index: int) -> List[str]:
        if sep_index >= len(separators) or len(text) <= max_chunk_size:
            return [text]
        
        sep = separators[sep_index]
        parts = text.split(sep)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(sep) <= max_chunk_size:
                if current_chunk:
                    current_chunk += sep + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.extend(split_recursively(current_chunk, sep_index + 1))
                current_chunk = part
        
        if current_chunk:
            chunks.extend(split_recursively(current_chunk, sep_index + 1))
        
        return chunks
    
    return split_recursively(document, 0)

def markdown_chunking(document: str) -> List[str]:
    """Split document at markdown headers.
    
    Args:
        document: Text in markdown format to split
    """
    chunks = []
    current_chunk = []
    
    for line in document.split('\n'):
        if line.startswith('#'):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_ollama_embedding_function(model_name: str):
    """Create a ChromaDB-compatible embedding function using Ollama."""
    class OllamaEmbeddingFunction:
        def __init__(self, model: str):
            self.model = model
            self.name = f"ollama:{model}"

        def __call__(self, texts: List[str]):
            embeddings_list = []
            for text in texts:
                # Call embeddings without options to let Ollama manage resources
                resp = ollama.embeddings(model=self.model, prompt=text)
                if isinstance(resp, dict) and "embedding" in resp:
                    embeddings_list.append(resp["embedding"])
                else:
                    try:
                        embeddings_list.append(resp[0]["embedding"])
                    except Exception:
                        raise RuntimeError("Unexpected Ollama embeddings response format")
            return embeddings_list

    return OllamaEmbeddingFunction(model_name)

def initialize_chroma_db(documents: List[str], batch_size: int = 10, progress_callback=None, document_metadata: dict | None = None, collection_name: str | None = None):
    """
    Initialize a ChromaDB collection with the provided documents.
    Tries upsert first with different parameter orders for compatibility,
    falls back to add if upsert fails.
    
    Args:
        documents: list of strings to add (already chunked as desired)
        batch_size: how many documents to process in each batch (default 10)
        progress_callback: optional function called after each batch. Preferred signature:
            progress_callback(current:int, total:int)
        For backward compatibility a single-float argument (ratio) is also supported.
        document_metadata: optional dict to attach as metadata to every document/chunk
    Returns:
        chroma collection object
    """
    # Use provided collection_name or default
    target_collection = collection_name if collection_name else COLLECTION_NAME

    try:
        import chromadb
        from chromadb.config import Settings

        # Initialize the persistent client pointing to CHROMA_PATH with telemetry disabled
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

        # Prepare the ollama-backed embedding function
        ollama_embed_fn = get_ollama_embedding_function(EMBEDDING_MODEL)

        print(f"Checking for existing collection '{target_collection}'...")
        try:
            # Try to get or create collection
            collection = client.get_or_create_collection(
                name=target_collection,
                embedding_function=ollama_embed_fn
            )
        except Exception as e:
            try:
                # Fallback: try to get existing collection first
                collection = client.get_collection(name=target_collection)
            except Exception:
                # If that fails, create new collection
                collection = client.create_collection(
                    name=target_collection,
                    embedding_function=ollama_embed_fn
                )

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate stable IDs for this batch
            batch_ids = [
                f"d_{hashlib.sha256(doc.encode('utf-8')).hexdigest()}" 
                for doc in batch
            ]
            
            # Prepare metadata for this batch
            batch_metadatas = [
                document_metadata.copy() if document_metadata else {}
                for _ in batch
            ]

            # Try to upsert the batch
            try:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch,
                    metadatas=batch_metadatas
                )
            except Exception:
                try:
                    # Fallback to add if upsert fails
                    collection.add(
                        documents=batch,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to add batch to collection: {e}")

            # Update progress if callback provided
            if progress_callback:
                current_chunk = i + len(batch)
                total_chunks = len(documents)
                try:
                    progress_callback(current_chunk, total_chunks)
                except TypeError:
                    # If the callback doesn't accept two arguments, fall back to progress ratio
                    progress_callback(float(current_chunk) / total_chunks)

        return collection

    except ImportError:
        print("ChromaDB not available - falling back to in-memory collection")
        
        # Fallback in-memory implementation
        class InMemoryCollection:
            def __init__(self, documents: List[str]):
                self.docs = []
                self.ids = []
                self.embeddings = []
                self.metadatas = []
                
                # Process documents in batches
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    for doc in batch:
                        # Add document
                        self.docs.append(doc)
                        
                        # Generate stable ID
                        doc_id = f"d_{hashlib.sha256(doc.encode('utf-8')).hexdigest()}"
                        self.ids.append(doc_id)
                        
                        # Generate embedding
                        try:
                            # emb_options = {"no_gpu": True, "device": "cpu", "num_gpu": 0}
                            emb_options = {
                                "num_gpu": 0,
                                "num_thread": 8 # Example: use 8 CPU threads
                                }
                            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc, options=emb_options)
                            if isinstance(resp, dict) and "embedding" in resp:
                                emb = resp["embedding"]
                            else:
                                emb = resp[0]["embedding"]
                            self.embeddings.append(emb)
                        except Exception as e:
                            raise RuntimeError(f"Failed to generate embedding: {e}")
                        
                        # Add metadata
                        self.metadatas.append(
                            document_metadata.copy() if document_metadata else {}
                        )
                    
                    # Update progress if callback provided
                    if progress_callback:
                        current_chunk = i + len(batch)
                        total_chunks = len(documents)
                        try:
                            progress_callback(current_chunk, total_chunks)
                        except TypeError:
                            # If the callback doesn't accept two arguments, fall back to progress ratio
                            progress_callback(float(current_chunk) / total_chunks)

            def query(self, query_texts: List[str], n_results: int = 3):
                # Only support single query for now
                query = query_texts[0]
                
                # Get query embedding
                try:
                    # emb_options = {"no_gpu": True, "device": "cpu", "num_gpu": 0}
                    emb_options = {
                            "num_gpu": 0,
                            "num_thread": 8 # Example: use 8 CPU threads
                        }
                    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query, options=emb_options)
                    if isinstance(resp, dict) and "embedding" in resp:
                        query_embedding = resp["embedding"]
                    else:
                        query_embedding = resp[0]["embedding"]
                except Exception as e:
                    raise RuntimeError(f"Failed to generate query embedding: {e}")

                # Cosine similarity function
                def cosine_similarity(a, b):
                    import math
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b))
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return dot / (norm_a * norm_b)

                # Calculate similarities and rank results
                similarities = [
                    cosine_similarity(query_embedding, doc_embedding) 
                    for doc_embedding in self.embeddings
                ]
                
                ranked_indices = sorted(
                    range(len(similarities)), 
                    key=lambda i: similarities[i],
                    reverse=True
                )[:n_results]
                
                # Gather results
                results = {
                    "documents": [[self.docs[i] for i in ranked_indices]],
                    "ids": [[self.ids[i] for i in ranked_indices]],
                    "metadatas": [[self.metadatas[i] for i in ranked_indices]],
                    "distances": [[1 - similarities[i] for i in ranked_indices]]
                }
                
                return results

        try:
            # Create and return in-memory collection
            collection = InMemoryCollection(documents)
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to create in-memory collection: {e}")