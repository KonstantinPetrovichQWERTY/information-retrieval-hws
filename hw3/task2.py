# Task 2:
# Copy and paste your KDTree class into a .py file.

from typing import List, Tuple, Optional, Union, Generator, Any, Dict
import numpy as np
import heapq

class KDTree:
    """
    A KD-Tree implementation for efficient nearest neighbor search.
    """

    def __init__(self, points: List[Tuple[int, np.ndarray]], dimension: int, distance_type: str = 'euclidean') -> None:
        """
        Initializes a new KD-Tree and selects the distance metric.

        Args:
            points: A list of (index, embedding) tuples to build the tree from.
            dimension: The dimensionality of the embedding vectors.
            distance_type: The type of distance metric to use ('euclidean' or 'cosine'). Defaults to 'euclidean'.
        """
        self.dimension: int = dimension
        self.root: Optional[Dict[str, Union[Tuple[int, np.ndarray], None, None]]] = None

        if distance_type == 'euclidean':
            self.distance_func = self._euclidean_distance
        elif distance_type == 'cosine':
            self.distance_func = self._cosine_distance
        else:
            raise ValueError("Invalid distance type. Use 'euclidean' or 'cosine'.")

        self.root = self._build_tree(points)

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    def _cosine_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))

    def _build_tree(self, points: List[Tuple[int, np.ndarray]], depth: int = 0) -> Optional[Dict[str, Union[Tuple[int, np.ndarray], Optional[Dict[str, Union[Tuple[int, np.ndarray], None, None]]]]]]:
        """
        Recursively builds the KD-Tree from the input points.

        Args:
            points: The set of points to build the tree from.
            depth: The current depth of the recursion, used to determine which dimension to split along.

        Returns:
            A node in the tree structure, containing information about the point and its child nodes.
        """
        if not points:
            return None

        axis = depth % self.dimension
        points.sort(key=lambda x: x[1][axis])
        median = len(points) // 2

        return {
            'point': points[median],
            'left': self._build_tree(points[:median], depth + 1),
            'right': self._build_tree(points[median + 1:], depth + 1)
        }

    def insert(self, new_point: Tuple[int, np.ndarray]) -> None:
        """
        Inserts a new point into the KD-Tree.

        Args:
            new_point: A tuple (index, embedding) to be added to the Tree.
        """
        def _insert(node, point, depth=0):
            if node is None:
                return {'point': point, 'left': None, 'right': None}

            axis = depth % self.dimension
            if point[1][axis] < node['point'][1][axis]:
                node['left'] = _insert(node['left'], point, depth + 1)
            else:
                node['right'] = _insert(node['right'], point, depth + 1)

            return node

        self.root = _insert(self.root, new_point)

    def find_knn(self, target: np.ndarray, k: int, include_distances: bool = True) -> List[Union[Tuple[float, Tuple[int, np.ndarray]], Tuple[int, np.ndarray]]]:
        """
        Finds the k-nearest neighbors to a target point in the KD-Tree.

        Args:
            target: The query embedding.
            k: Number of nearest neighbors to look up.
            include_distances: Whether to return distances between query and neighbors. Default is True.

        Returns:
            List of k-nearest neighbors and optionally distances to those neighbors.
        """
        max_heap = []
        self._search_knn(self.root, target, k, max_heap)

        nearest_neighbors = heapq.nsmallest(k, max_heap)
        if include_distances:
            return [(dist, point) for dist, point in nearest_neighbors]
        else:
            return [point for dist, point in nearest_neighbors]

    def _search_knn(self, current_node: Optional[Dict[str, Any]],
                    target_point: np.ndarray, k: int,
                    max_heap: List[Tuple[float, Tuple[int, np.ndarray]]],
                    depth: int = 0) -> None:
        """
        Recursively searches the KD-Tree for the k-nearest neighbors.

        This method uses a max-heap to efficiently track the k closest points found so far.

        Args:
            current_node: The current node being visited (dictionary with 'point', 'left', 'right').
            target_point: The query point.
            k: The number of nearest neighbors to find.
            max_heap: A max-heap (using heapq) storing (-distance, (index, point)).
            depth: Recursion depth (used for splitting dimension).
        """
        if current_node is None:
            return

        point = current_node['point']
        distance = self.distance_func(point[1], target_point)

        if len(max_heap) < k:
            heapq.heappush(max_heap, (-distance, point))
        elif distance < -max_heap[0][0]:
            heapq.heappushpop(max_heap, (-distance, point))

        axis = depth % self.dimension
        if target_point[axis] < point[1][axis]:
            self._search_knn(current_node['left'], target_point, k, max_heap, depth + 1)
            if len(max_heap) < k or abs(target_point[axis] - point[1][axis]) < -max_heap[0][0]:
                self._search_knn(current_node['right'], target_point, k, max_heap, depth + 1)
        else:
            self._search_knn(current_node['right'], target_point, k, max_heap, depth + 1)
            if len(max_heap) < k or abs(target_point[axis] - point[1][axis]) < -max_heap[0][0]:
                self._search_knn(current_node['left'], target_point, k, max_heap, depth + 1)

    def nearest_neighbor(self, target_point: np.ndarray, k: int = 5, include_distance: bool = True) -> Optional[List[Union[Tuple[float, Tuple[int, np.ndarray]], Tuple[int, np.ndarray]]]]:
        """
        Finds the nearest neighbor to a target point by calling find_knn and returning the result up to k.

        Args:
            target: The query embedding.
            k: Number of nearest neighbors to look up.
            include_distances: Whether to return distances. Default is True.

        Returns:
            Optional list of the nearest points and optionally distances.
        """
        return self.find_knn(target_point, k, include_distance)

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterates through all stored embeddings with their indices.

        Returns:
            A generator yielding (index, embedding) tuples.
        """
        def _traverse(node):
            if node is None:
                return
            yield node['point']
            yield from _traverse(node['left'])
            yield from _traverse(node['right'])

        return _traverse(self.root)

    def range_query(self, target: Union[np.ndarray, Tuple[int, np.ndarray]], radius: float) -> List[int]:
        """
        Finds all points within a certain radius from the target point.

        Args:
            target_point: The query embedding.
            radius: The maximum allowable distance from the target point.

        Returns:
            A list of indices within the radius.
        """
        results = []

        def _recursive_search(node, target_embedding, depth=0):
            if node is None:
                return

            point = node['point']
            distance = self.distance_func(point[1], target_embedding)

            if distance <= radius:
                results.append(point[0])

            axis = depth % self.dimension
            if target_embedding[axis] - radius < point[1][axis]:
                _recursive_search(node['left'], target_embedding, depth + 1)
            if target_embedding[axis] + radius > point[1][axis]:
                _recursive_search(node['right'], target_embedding, depth + 1)

        _recursive_search(self.root, target[1] if isinstance(target, tuple) else target)
        return results

