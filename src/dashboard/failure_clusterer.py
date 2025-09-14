"""ML clustering for test failures using scikit-learn and PyTorch."""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class FailureCluster:
    """Represents a cluster of similar test failures."""
    cluster_id: int
    size: int
    centroid: np.ndarray
    failures: List[Dict[str, Any]]
    common_patterns: List[str]
    representative_failure: Dict[str, Any]
    confidence_score: float


@dataclass
class ClusteringResult:
    """Result of failure clustering analysis."""
    clusters: List[FailureCluster]
    silhouette_score: float
    optimal_clusters: int
    method_used: str
    feature_importance: Dict[str, float]


class FailureClusterer:
    """Clusters test failures using various ML techniques."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
    
    def cluster_failures(self, failures: List[Dict[str, Any]], 
                        method: str = 'auto') -> ClusteringResult:
        """Cluster test failures using specified method."""
        
        try:
            if not failures:
                return ClusteringResult(
                    clusters=[],
                    silhouette_score=0.0,
                    optimal_clusters=0,
                    method_used='none',
                    feature_importance={}
                )
            
            # Prepare features for clustering
            features = self._prepare_features(failures)
            
            # Determine optimal clustering method and parameters
            if method == 'auto':
                method, n_clusters = self._find_optimal_clustering(features, failures)
            else:
                n_clusters = self._estimate_clusters(features)
            
            # Perform clustering
            if method == 'kmeans':
                clusters, labels = self._kmeans_clustering(features, n_clusters)
            elif method == 'dbscan':
                clusters, labels = self._dbscan_clustering(features)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            elif method == 'hierarchical':
                clusters, labels = self._hierarchical_clustering(features, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Create cluster objects
            failure_clusters = self._create_cluster_objects(failures, labels, features)
            
            # Calculate silhouette score
            if len(set(labels)) > 1 and -1 not in labels:
                silhouette = silhouette_score(features, labels)
            else:
                silhouette = 0.0
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(features, labels)
            
            result = ClusteringResult(
                clusters=failure_clusters,
                silhouette_score=silhouette,
                optimal_clusters=n_clusters,
                method_used=method,
                feature_importance=feature_importance
            )
            
            logger.info(f"Clustered {len(failures)} failures into {len(failure_clusters)} clusters using {method}")
            return result
            
        except Exception as e:
            logger.error(f"Error clustering failures: {e}")
            return ClusteringResult(
                clusters=[],
                silhouette_score=0.0,
                optimal_clusters=0,
                method_used='error',
                feature_importance={}
            )
    
    def _prepare_features(self, failures: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for clustering from failure data."""
        
        try:
            # Extract text features
            texts = []
            for failure in failures:
                text = f"{failure.get('error_message', '')} {failure.get('stack_trace', '')} {failure.get('test_name', '')}"
                texts.append(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Create additional features
            additional_features = []
            for failure in failures:
                features = [
                    len(failure.get('error_message', '')),
                    len(failure.get('stack_trace', '')),
                    failure.get('duration', 0),
                    1 if 'timeout' in failure.get('error_message', '').lower() else 0,
                    1 if 'assertion' in failure.get('error_message', '').lower() else 0,
                    1 if 'network' in failure.get('error_message', '').lower() else 0,
                    1 if 'database' in failure.get('error_message', '').lower() else 0,
                ]
                additional_features.append(features)
            
            # Combine embeddings with additional features
            additional_features = np.array(additional_features)
            combined_features = np.hstack([embeddings, additional_features])
            
            # Normalize features
            combined_features = self.scaler.fit_transform(combined_features)
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return simple features as fallback
            return np.random.rand(len(failures), 10)
    
    def _find_optimal_clustering(self, features: np.ndarray, 
                               failures: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Find optimal clustering method and parameters."""
        
        try:
            # Try different clustering methods
            methods = ['kmeans', 'dbscan', 'hierarchical']
            best_score = -1
            best_method = 'kmeans'
            best_clusters = 3
            
            # Estimate number of clusters
            n_clusters = self._estimate_clusters(features)
            
            for method in methods:
                try:
                    if method == 'kmeans':
                        clusters, labels = self._kmeans_clustering(features, n_clusters)
                    elif method == 'dbscan':
                        clusters, labels = self._dbscan_clustering(features)
                    elif method == 'hierarchical':
                        clusters, labels = self._hierarchical_clustering(features, n_clusters)
                    
                    # Calculate silhouette score
                    if len(set(labels)) > 1 and -1 not in labels:
                        score = silhouette_score(features, labels)
                        if score > best_score:
                            best_score = score
                            best_method = method
                            best_clusters = len(set(labels))
                    
                except Exception as e:
                    logger.warning(f"Error with {method} clustering: {e}")
                    continue
            
            return best_method, best_clusters
            
        except Exception as e:
            logger.error(f"Error finding optimal clustering: {e}")
            return 'kmeans', 3
    
    def _estimate_clusters(self, features: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method."""
        
        try:
            if len(features) < 4:
                return 2
            
            max_clusters = min(len(features) // 2, 10)
            if max_clusters < 2:
                return 2
            
            inertias = []
            K_range = range(2, max_clusters + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            if len(inertias) >= 2:
                # Calculate second derivative to find elbow
                second_deriv = np.diff(np.diff(inertias))
                elbow_idx = np.argmax(second_deriv) + 2  # +2 because of double diff
                optimal_k = K_range[min(elbow_idx, len(K_range) - 1)]
            else:
                optimal_k = 2
            
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error estimating clusters: {e}")
            return 3
    
    def _kmeans_clustering(self, features: np.ndarray, n_clusters: int) -> Tuple[Any, np.ndarray]:
        """Perform K-means clustering."""
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return kmeans, labels
    
    def _dbscan_clustering(self, features: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Perform DBSCAN clustering."""
        
        # Use PCA to reduce dimensionality for DBSCAN
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        labels = dbscan.fit_predict(features_reduced)
        
        return dbscan, labels
    
    def _hierarchical_clustering(self, features: np.ndarray, n_clusters: int) -> Tuple[Any, np.ndarray]:
        """Perform hierarchical clustering."""
        
        # Use PCA to reduce dimensionality for hierarchical clustering
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(features_reduced)
        
        return hierarchical, labels
    
    def _create_cluster_objects(self, failures: List[Dict[str, Any]], 
                              labels: np.ndarray, 
                              features: np.ndarray) -> List[FailureCluster]:
        """Create FailureCluster objects from clustering results."""
        
        clusters = []
        unique_labels = set(labels)
        
        # Remove noise label (-1) if present
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in unique_labels:
            # Get failures in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_failures = [failures[i] for i in cluster_indices]
            
            if not cluster_failures:
                continue
            
            # Calculate centroid
            cluster_features = features[cluster_indices]
            centroid = np.mean(cluster_features, axis=0)
            
            # Find representative failure (closest to centroid)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            representative_failure = failures[representative_idx]
            
            # Extract common patterns
            common_patterns = self._extract_common_patterns(cluster_failures)
            
            # Calculate confidence score
            confidence_score = self._calculate_cluster_confidence(cluster_features, centroid)
            
            cluster = FailureCluster(
                cluster_id=cluster_id,
                size=len(cluster_failures),
                centroid=centroid,
                failures=cluster_failures,
                common_patterns=common_patterns,
                representative_failure=representative_failure,
                confidence_score=confidence_score
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _extract_common_patterns(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from failures in a cluster."""
        
        patterns = []
        
        try:
            # Extract error types
            error_types = [f.get('error_type', 'Unknown') for f in failures]
            common_error_types = self._find_common_elements(error_types)
            patterns.extend([f"Error type: {et}" for et in common_error_types])
            
            # Extract common keywords
            all_messages = ' '.join([f.get('error_message', '') for f in failures])
            common_keywords = self._extract_keywords(all_messages)
            patterns.extend([f"Keyword: {kw}" for kw in common_keywords[:5]])
            
            # Extract common test patterns
            test_names = [f.get('test_name', '') for f in failures]
            common_test_patterns = self._find_common_elements(test_names)
            patterns.extend([f"Test pattern: {tp}" for tp in common_test_patterns[:3]])
            
        except Exception as e:
            logger.error(f"Error extracting common patterns: {e}")
        
        return patterns[:10]  # Limit to 10 patterns
    
    def _find_common_elements(self, items: List[str], threshold: float = 0.3) -> List[str]:
        """Find elements that appear in more than threshold of items."""
        
        from collections import Counter
        
        counter = Counter(items)
        total = len(items)
        threshold_count = int(total * threshold)
        
        return [item for item, count in counter.items() if count >= threshold_count]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        
        try:
            # Simple keyword extraction
            words = text.lower().split()
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            
            from collections import Counter
            counter = Counter(words)
            return [word for word, count in counter.most_common(10)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _calculate_cluster_confidence(self, cluster_features: np.ndarray, 
                                    centroid: np.ndarray) -> float:
        """Calculate confidence score for a cluster."""
        
        try:
            # Calculate average distance from centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            avg_distance = np.mean(distances)
            
            # Normalize confidence (lower distance = higher confidence)
            confidence = 1.0 / (1.0 + avg_distance)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating cluster confidence: {e}")
            return 0.5
    
    def _calculate_feature_importance(self, features: np.ndarray, 
                                    labels: np.ndarray) -> Dict[str, float]:
        """Calculate importance of different features for clustering."""
        
        try:
            # Simple feature importance based on variance
            feature_vars = np.var(features, axis=0)
            total_var = np.sum(feature_vars)
            
            importance = {}
            feature_names = [f"feature_{i}" for i in range(len(feature_vars))]
            
            for name, var in zip(feature_names, feature_vars):
                importance[name] = var / total_var if total_var > 0 else 0.0
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def visualize_clusters(self, clusters: List[FailureCluster], 
                          output_path: str) -> bool:
        """Visualize clusters (placeholder for future implementation)."""
        
        try:
            # This would generate visualization plots
            # For now, just log the cluster information
            logger.info(f"Visualizing {len(clusters)} clusters to {output_path}")
            
            for cluster in clusters:
                logger.info(f"Cluster {cluster.cluster_id}: {cluster.size} failures, "
                          f"confidence: {cluster.confidence_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing clusters: {e}")
            return False
