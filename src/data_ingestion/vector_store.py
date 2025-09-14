"""Vector store management for indexing and retrieving documents."""

import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations for document indexing and retrieval."""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = None
        self.faiss_index = None
        self._initialize_chroma()
        self._initialize_faiss()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client."""
        try:
            # Ensure directory exists
            os.makedirs(self.settings.chroma_persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="test_automation_docs",
                metadata={"description": "Documents for AI test automation"}
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
    
    def _initialize_faiss(self):
        """Initialize FAISS index for similarity search."""
        try:
            faiss_path = self.settings.faiss_index_path
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
            
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                logger.info("FAISS index loaded from disk")
            else:
                # Create new index with 384 dimensions (all-MiniLM-L6-v2 output size)
                self.faiss_index = faiss.IndexFlatIP(384)
                logger.info("New FAISS index created")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.faiss_index = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to both ChromaDB and FAISS."""
        if not documents:
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Add to ChromaDB
            if self.chroma_client:
                self.collection.add(
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} documents to ChromaDB")
            
            # Add to FAISS
            if self.faiss_index:
                embeddings = self.embedding_model.encode(contents)
                # Normalize embeddings for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Save FAISS index
                self._save_faiss_index()
                
                logger.info(f"Added {len(documents)} documents to FAISS")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_similar(self, query: str, n_results: int = 5, 
                      metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            results = []
            
            # Search in ChromaDB
            if self.chroma_client:
                chroma_results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=metadata_filter
                )
                
                for i in range(len(chroma_results['ids'][0])):
                    result = {
                        'id': chroma_results['ids'][0][i],
                        'content': chroma_results['documents'][0][i],
                        'metadata': chroma_results['metadatas'][0][i],
                        'distance': chroma_results['distances'][0][i]
                    }
                    results.append(result)
            
            # Search in FAISS for additional results
            if self.faiss_index and len(results) < n_results:
                faiss_results = self.faiss_index.search(
                    query_embedding.astype('float32'), 
                    min(n_results, self.faiss_index.ntotal)
                )
                
                # Note: FAISS results would need document mapping
                # This is a simplified implementation
                
            # Sort by similarity score (lower distance = more similar)
            results.sort(key=lambda x: x['distance'])
            
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Search documents by metadata filters."""
        try:
            if not self.chroma_client:
                return []
            
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            if self.chroma_client:
                self.collection.delete(ids=document_ids)
                logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            
            # Note: FAISS deletion is more complex and would require rebuilding
            # For now, we'll skip FAISS deletion
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            stats = {
                'chroma_count': 0,
                'faiss_count': 0,
                'total_documents': 0
            }
            
            if self.chroma_client:
                chroma_count = self.collection.count()
                stats['chroma_count'] = chroma_count
            
            if self.faiss_index:
                stats['faiss_count'] = self.faiss_index.ntotal
            
            stats['total_documents'] = max(stats['chroma_count'], stats['faiss_count'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        try:
            if self.faiss_index:
                faiss.write_index(self.faiss_index, self.settings.faiss_index_path)
                logger.debug("FAISS index saved to disk")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def reset_collection(self):
        """Reset the entire collection (use with caution)."""
        try:
            if self.chroma_client:
                self.chroma_client.delete_collection("test_automation_docs")
                self.collection = self.chroma_client.create_collection(
                    name="test_automation_docs"
                )
                logger.info("ChromaDB collection reset")
            
            if self.faiss_index:
                self.faiss_index = faiss.IndexFlatIP(384)
                self._save_faiss_index()
                logger.info("FAISS index reset")
                
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
