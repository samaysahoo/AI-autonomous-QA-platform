#!/usr/bin/env python3
"""Initialize vector store with sample data."""

import sys
import os
import asyncio
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.data_ingestion.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_vector_store():
    """Initialize vector store with sample data."""
    
    logger.info("Initializing vector store...")
    
    try:
        vector_store = VectorStoreManager()
        
        # Sample documents for initialization
        sample_docs = [
            {
                "id": "doc_1",
                "content": """
                User Login Functionality
                As a user, I want to be able to log in to the application using my email and password
                so that I can access my personal dashboard and features.
                
                Acceptance Criteria:
                - User can enter email address
                - User can enter password
                - System validates credentials
                - User is redirected to dashboard on successful login
                - Error message displayed for invalid credentials
                """,
                "metadata": {
                    "source": "jira",
                    "type": "User Story",
                    "priority": "High",
                    "labels": ["authentication", "login", "user-management"],
                    "components": ["authentication"]
                }
            },
            {
                "id": "doc_2",
                "content": """
                Payment Processing Error
                Users are experiencing payment failures when trying to complete transactions.
                The error occurs intermittently and appears to be related to network timeouts
                during the payment gateway communication.
                
                Stack Trace:
                PaymentGatewayException: Connection timeout after 30 seconds
                at PaymentService.processPayment(PaymentService.java:45)
                at CheckoutController.handlePayment(CheckoutController.java:123)
                
                Impact: High - affects revenue and user experience
                """,
                "metadata": {
                    "source": "analytics",
                    "error_type": "PaymentGatewayException",
                    "severity": "High",
                    "affected_components": ["payment", "checkout"],
                    "frequency": 15
                }
            },
            {
                "id": "doc_3",
                "content": """
                Test Case: User Registration Flow
                Test Steps:
                1. Navigate to registration page
                2. Enter valid email address
                3. Enter password meeting requirements
                4. Confirm password
                5. Click register button
                6. Verify email confirmation sent
                7. Verify user account created
                
                Expected Result: User account created successfully and confirmation email sent
                """,
                "metadata": {
                    "source": "test",
                    "type": "Test Case",
                    "framework": "appium",
                    "priority": "Medium",
                    "tags": ["registration", "user-management", "e2e"]
                }
            },
            {
                "id": "doc_4",
                "content": """
                Database Connection Pool Exhaustion
                Application is experiencing database connection pool exhaustion during peak hours.
                This leads to slow response times and occasional service unavailability.
                
                Symptoms:
                - Slow database queries
                - Connection timeout errors
                - High memory usage
                - Service degradation during peak hours
                
                Root Cause: Insufficient connection pool size for current load
                """,
                "metadata": {
                    "source": "analytics",
                    "error_type": "DatabaseConnectionException",
                    "severity": "Critical",
                    "affected_components": ["database", "performance"],
                    "frequency": 8
                }
            },
            {
                "id": "doc_5",
                "content": """
                Mobile App UI Test Scenarios
                Test the mobile application user interface across different devices and screen sizes.
                
                Test Cases:
                1. Login screen layout on different screen sizes
                2. Navigation menu functionality
                3. Form validation and error handling
                4. Button interactions and touch responses
                5. Text readability and accessibility
                
                Framework: Appium
                Devices: iOS and Android
                """,
                "metadata": {
                    "source": "test",
                    "type": "Test Specification",
                    "framework": "appium",
                    "priority": "High",
                    "tags": ["mobile", "ui", "cross-platform"]
                }
            }
        ]
        
        # Add documents to vector store
        success = vector_store.add_documents(sample_docs)
        
        if success:
            logger.info(f"Successfully added {len(sample_docs)} sample documents to vector store")
            
            # Get collection stats
            stats = vector_store.get_collection_stats()
            logger.info(f"Vector store stats: {stats}")
            
            # Test search functionality
            test_queries = [
                "user login authentication",
                "payment processing errors",
                "database connection issues",
                "mobile app testing"
            ]
            
            for query in test_queries:
                results = vector_store.search_similar(query, n_results=3)
                logger.info(f"Search '{query}': found {len(results)} results")
                
                if results:
                    top_result = results[0]
                    logger.info(f"  Top result: {top_result['id']} (distance: {top_result['distance']:.3f})")
            
        else:
            logger.error("Failed to add documents to vector store")
            return False
        
        logger.info("Vector store initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(initialize_vector_store())
    if success:
        print("Vector store initialized successfully!")
    else:
        print("Vector store initialization failed!")
        sys.exit(1)
