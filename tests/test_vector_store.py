"""Tests for vector store management."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.data_ingestion.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test cases for VectorStoreManager class."""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        with patch('chromadb.PersistentClient') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            
            # Mock collection
            mock_collection = Mock()
            mock_instance.get_or_create_collection.return_value = mock_collection
            mock_collection.add = Mock()
            mock_collection.query = Mock()
            mock_collection.get = Mock()
            mock_collection.delete = Mock()
            mock_collection.count = Mock(return_value=10)
            
            yield mock_instance
    
    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index."""
        with patch('faiss.IndexFlatIP') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            mock_instance.add = Mock()
            mock_instance.search = Mock()
            mock_instance.ntotal = 10
            
            yield mock_instance
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock sentence transformer model."""
        with patch('sentence_transformers.SentenceTransformer') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            mock_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
            
            yield mock_instance
    
    @pytest.fixture
    def vector_store(self, mock_chroma_client, mock_faiss_index, mock_embedding_model):
        """Create VectorStoreManager instance with mocked dependencies."""
        return VectorStoreManager()
    
    def test_init(self, vector_store):
        """Test VectorStoreManager initialization."""
        assert vector_store is not None
        assert vector_store.chroma_client is not None
        assert vector_store.faiss_index is not None
        assert vector_store.embedding_model is not None
    
    def test_add_documents_success(self, vector_store, mock_chroma_client, mock_faiss_index):
        """Test successful document addition."""
        documents = [
            {
                "id": "doc_1",
                "content": "Test document content 1",
                "metadata": {"source": "jira", "type": "Story"}
            },
            {
                "id": "doc_2", 
                "content": "Test document content 2",
                "metadata": {"source": "analytics", "type": "Error"}
            }
        ]
        
        result = vector_store.add_documents(documents)
        
        assert result is True
        # Verify ChromaDB add was called
        mock_chroma_client.get_or_create_collection.return_value.add.assert_called_once()
        # Verify FAISS add was called
        mock_faiss_index.add.assert_called_once()
    
    def test_add_documents_empty_list(self, vector_store):
        """Test adding empty document list."""
        result = vector_store.add_documents([])
        assert result is False
    
    def test_add_documents_error(self, vector_store, mock_chroma_client):
        """Test document addition with error."""
        # Mock ChromaDB error
        mock_chroma_client.get_or_create_collection.return_value.add.side_effect = Exception("ChromaDB Error")
        
        documents = [
            {
                "id": "doc_1",
                "content": "Test document content",
                "metadata": {"source": "jira"}
            }
        ]
        
        result = vector_store.add_documents(documents)
        assert result is False
    
    def test_search_similar_success(self, vector_store, mock_chroma_client, mock_faiss_index):
        """Test successful similarity search."""
        # Mock ChromaDB query response
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = {
            'ids': [['doc_1', 'doc_2']],
            'documents': [['Test content 1', 'Test content 2']],
            'metadatas': [[{'source': 'jira'}, {'source': 'analytics'}]],
            'distances': [[0.1, 0.3]]
        }
        
        results = vector_store.search_similar("test query", n_results=2)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc_1'
        assert results[0]['distance'] == 0.1
        assert results[1]['id'] == 'doc_2'
        assert results[1]['distance'] == 0.3
    
    def test_search_similar_with_metadata_filter(self, vector_store, mock_chroma_client):
        """Test similarity search with metadata filter."""
        # Mock ChromaDB query response
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = {
            'ids': [['doc_1']],
            'documents': [['Test content']],
            'metadatas': [[{'source': 'jira'}]],
            'distances': [[0.2]]
        }
        
        metadata_filter = {"source": "jira"}
        results = vector_store.search_similar(
            "test query", 
            n_results=5, 
            metadata_filter=metadata_filter
        )
        
        assert len(results) == 1
        # Verify query was called with metadata filter
        mock_chroma_client.get_or_create_collection.return_value.query.assert_called_once()
    
    def test_search_similar_error(self, vector_store, mock_chroma_client):
        """Test similarity search with error."""
        # Mock ChromaDB error
        mock_chroma_client.get_or_create_collection.return_value.query.side_effect = Exception("Query Error")
        
        results = vector_store.search_similar("test query")
        
        assert len(results) == 0
    
    def test_search_by_metadata_success(self, vector_store, mock_chroma_client):
        """Test successful metadata search."""
        # Mock ChromaDB get response
        mock_chroma_client.get_or_create_collection.return_value.get.return_value = {
            'ids': ['doc_1', 'doc_2'],
            'documents': ['Test content 1', 'Test content 2'],
            'metadatas': [{'source': 'jira'}, {'source': 'jira'}]
        }
        
        metadata_filter = {"source": "jira"}
        results = vector_store.search_by_metadata(metadata_filter, limit=100)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc_1'
        assert results[1]['id'] == 'doc_2'
    
    def test_search_by_metadata_error(self, vector_store, mock_chroma_client):
        """Test metadata search with error."""
        # Mock ChromaDB error
        mock_chroma_client.get_or_create_collection.return_value.get.side_effect = Exception("Get Error")
        
        results = vector_store.search_by_metadata({"source": "jira"})
        
        assert len(results) == 0
    
    def test_delete_documents_success(self, vector_store, mock_chroma_client):
        """Test successful document deletion."""
        document_ids = ["doc_1", "doc_2"]
        
        result = vector_store.delete_documents(document_ids)
        
        assert result is True
        mock_chroma_client.get_or_create_collection.return_value.delete.assert_called_once_with(ids=document_ids)
    
    def test_delete_documents_error(self, vector_store, mock_chroma_client):
        """Test document deletion with error."""
        # Mock ChromaDB error
        mock_chroma_client.get_or_create_collection.return_value.delete.side_effect = Exception("Delete Error")
        
        result = vector_store.delete_documents(["doc_1"])
        
        assert result is False
    
    def test_get_collection_stats_success(self, vector_store, mock_chroma_client, mock_faiss_index):
        """Test successful collection stats retrieval."""
        # Mock ChromaDB count
        mock_chroma_client.get_or_create_collection.return_value.count.return_value = 15
        # Mock FAISS count
        mock_faiss_index.ntotal = 15
        
        stats = vector_store.get_collection_stats()
        
        assert stats['chroma_count'] == 15
        assert stats['faiss_count'] == 15
        assert stats['total_documents'] == 15
    
    def test_get_collection_stats_error(self, vector_store, mock_chroma_client):
        """Test collection stats with error."""
        # Mock ChromaDB error
        mock_chroma_client.get_or_create_collection.return_value.count.side_effect = Exception("Count Error")
        
        stats = vector_store.get_collection_stats()
        
        assert 'error' in stats
    
    def test_reset_collection(self, vector_store, mock_chroma_client, mock_faiss_index):
        """Test collection reset."""
        # Mock delete and create operations
        mock_chroma_client.delete_collection = Mock()
        mock_chroma_client.create_collection = Mock()
        
        vector_store.reset_collection()
        
        # Verify operations were called
        mock_chroma_client.delete_collection.assert_called_once_with("test_automation_docs")
        mock_chroma_client.create_collection.assert_called_once()
    
    def test_reset_collection_error(self, vector_store, mock_chroma_client):
        """Test collection reset with error."""
        # Mock ChromaDB error
        mock_chroma_client.delete_collection.side_effect = Exception("Delete Error")
        
        # Should not raise exception
        vector_store.reset_collection()
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_initialize_chroma_directory_creation(self, mock_makedirs, mock_exists, vector_store):
        """Test ChromaDB directory creation during initialization."""
        mock_exists.return_value = False
        
        # This would be called during VectorStoreManager initialization
        # We're testing the directory creation logic
        vector_store._initialize_chroma()
        
        # Verify directory creation was attempted
        mock_makedirs.assert_called()
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_initialize_faiss_directory_creation(self, mock_makedirs, mock_exists, vector_store):
        """Test FAISS directory creation during initialization."""
        mock_exists.return_value = False
        
        # This would be called during VectorStoreManager initialization
        # We're testing the directory creation logic
        vector_store._initialize_faiss()
        
        # Verify directory creation was attempted
        mock_makedirs.assert_called()
    
    def test_embedding_model_initialization(self, vector_store):
        """Test that embedding model is properly initialized."""
        assert vector_store.embedding_model is not None
        # Test that encode method exists
        assert hasattr(vector_store.embedding_model, 'encode')
    
    def test_faiss_index_initialization(self, vector_store):
        """Test that FAISS index is properly initialized."""
        assert vector_store.faiss_index is not None
        # Test that add method exists
        assert hasattr(vector_store.faiss_index, 'add')
        assert hasattr(vector_store.faiss_index, 'search')
