# Task 3:
# Copy and paste your LSH class into a .py file.

import numpy as np
from typing import List, Tuple

class LSH:
    def __init__(self, index_data: np.ndarray, bucket_size: int = 16, seed: int = 0, distance_type: str = 'cosine'):
        
        """
        Initialize LSH with data, bucket size, random seed, and distance type.

        :param index_data: Array of tuples where each tuple consists of an index and data point.
        :param bucket_size: Number of data points per bucket.
        :param seed: Seed for random number generator.
        :param distance_type: Type of distance metric, either 'euclidean' or 'cosine'.
        """

        self.indices, self.data = zip(*index_data)
        self.data = np.asarray(self.data)
        self.bucket_size = bucket_size
        self.rng = np.random.default_rng(seed)
        self.hyperplanes = self._generate_hyperplanes()
        self.hash_table = self._create_hash_table()

        if distance_type == 'euclidean':
            self.distance_func = self._euclidean_distance
        elif distance_type == 'cosine':
            self.distance_func = self._cosine_distance
        else:
            raise ValueError("Invalid distance type. Use 'euclidean' or 'cosine'.")

    def _generate_hyperplanes(self) -> np.ndarray:
        
        """
        Generate random hyperplanes for hashing based on feature dimensions and bucket size.

        :return: Array of hyperplanes for hashing data.
        """

        feature_dim = self.data.shape[1]
        num_hyperplanes = int(np.ceil(np.log2(len(self.data) / self.bucket_size)))
        return self.rng.normal(size=(num_hyperplanes, feature_dim))

    def _generate_hash_key(self, points: np.ndarray) -> np.ndarray:
        
        """
        Generate a hash key for given points based on the hyperplanes
        Remember that you need to convert the resulting binary hash into a decimal value.

        :param points: Array of data points to hash.
        :return: Hash keys for the data points.
        """

        projections = np.dot(points, self.hyperplanes.T)
        binary_hash = (projections > 0).astype(int)
        decimal_hash = binary_hash.dot(1 << np.arange(binary_hash.shape[1])[::-1])  # Convert binary to decimal
        return decimal_hash

    def _query_hash_candidates(self, query: np.ndarray, repeat: int = 10) -> List[int]:
        
        """
        Retrieve candidates from hash table based on query and specified repeat count.

        :param query: Query data point.
        :param repeat: Number of times to hash the query for candidate retrieval.
        :return: List of candidate indices.
        """

        candidates = set()
        for _ in range(repeat):
            hash_key = self._generate_hash_key(query.reshape(1, -1))[0]
            candidates.update(self.hash_table.get(hash_key, []))
        return list(candidates)

    def _euclidean_distance(self, points: np.ndarray, query: np.ndarray) -> np.ndarray:
        
        """
        Compute Euclidean distance between points and query.

        :param points: Array of points to compare.
        :param query: Query point.
        :return: Array of distances.
        """

        return np.linalg.norm(points - query, axis=1)

    def _cosine_distance(self, points: np.ndarray, query: np.ndarray) -> np.ndarray:
        
        """
        Compute Cosine distance between points and query.

        :param points: Array of points to compare.
        :param query: Query point.
        :return: Array of cosine distances.
        """

        query_norm = np.linalg.norm(query)
        points_norm = np.linalg.norm(points, axis=1)
        cosine_similarity = np.dot(points, query) / (points_norm * query_norm)
        return 1 - cosine_similarity  # Cosine distance = 1 - similarity

    def _create_hash_table(self) -> dict:
        
        """
        Create a hash table for the LSH algorithm by mapping data points to hash buckets.

        :return: Hash table with keys as hash values and values as lists of data indices.
        """

        hash_table = {}
        hash_keys = self._generate_hash_key(self.data)
        for idx, key in zip(self.indices, hash_keys):
            if key not in hash_table:
                hash_table[key] = []
            hash_table[key].append(idx)
        return hash_table

    def approximate_knn_search(self, query: np.ndarray, k: int = 5, repeat: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Perform approximate K-nearest neighbor search on the query point.

        :param query: Query point for which nearest neighbors are sought.
        :param k: Number of neighbors to retrieve.
        :param repeat: Number of times to hash the query to increase candidate count.
        :return: Tuple of nearest points, their distances, and their original indices.
        """

        candidates_indices = self._query_hash_candidates(query, repeat=repeat)
        if not candidates_indices:
            return np.array([]), np.array([]), np.array([])  # Return empty arrays if no candidates

        candidates = self.data[list(candidates_indices)]
        distances = self.distance_func(candidates, query)
        sorted_indices = np.argsort(distances)[:k]
        nearest_points = candidates[sorted_indices]
        nearest_distances = distances[sorted_indices]
        nearest_indices = np.array(candidates_indices)[sorted_indices]

        return nearest_points, nearest_distances, nearest_indices
