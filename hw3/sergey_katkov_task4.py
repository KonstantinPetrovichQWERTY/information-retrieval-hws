# Task 4:

# Copy all the explanations you wrote in the notebook and include them as comments in a .py file.
# Copy all the code from the notebook’s code cells into the same file, with proper comments.
# Submit your complete notebook as firstname_lastname.ipynb.

# In total, you will submit 5 files.
# File Naming Convention
# Name the files as follows:

# firstname_familyname_task1.py (for Task 1)
# The naming must correspond to the format of your email.
# For example, if your name is Mark Ivan (as in m.ivan@innopolis.university):
# File for Task 1: mark_ivan_task1.py.
# similar to all other tasks


"""
Write you explanation here

To achieve cross-modal retrieval, we need to project the image and text embeddings into a common space. One effective way to do this is by using Canonical Correlation Analysis (CCA).

Apply CCA:

Use CCA to project both image and text embeddings into a shared space.
Build KDTree:
Build a KDTree for both image and text embeddings in the shared space.
Perform Cross-Modal Retrieval:
For a text query, project it into the shared space and use the KDTree to find the nearest image embeddings.
For an image query, project it into the shared space and use the KDTree to find the nearest text embeddings.
"""

import numpy as np
from sklearn.cross_decomposition import CCA

# Extract embeddings from the lists
image_embeddings = np.array([emb for _, emb in image_data])
text_embeddings = np.array([emb for _, emb in text_data])

# Apply CCA to project both embeddings into a shared space
cca = CCA(n_components=384)
cca.fit(image_embeddings, text_embeddings)

# Project both image and text embeddings into the shared space
image_embeddings_cca = cca.transform(image_embeddings)
text_embeddings_cca = cca.transform(text_embeddings)

image_embeddings_cca_indexed = [(index, img) for index, img in enumerate(image_embeddings_cca)]
text_embeddings_cca_indexed = [(index, txt) for index, txt in enumerate(text_embeddings_cca)]

# Build KDTree for fast nearest neighbor search
image_tree = KDTree(image_embeddings_cca_indexed, len(image_embeddings_cca_indexed[0][1]))
text_tree = KDTree(text_embeddings_cca_indexed, len(text_embeddings_cca_indexed[0][1]))

def text_to_image_retrieval(text_query_embedding, top_k=10):
    text_query_cca = cca.transform([text_query_embedding])[0]
    nearest_neighbors = image_tree.nearest_neighbor(text_query_cca, k=top_k, include_distance=False)
    return nearest_neighbors

# Function to perform image-to-text retrieval
def image_to_text_retrieval(image_query_embedding, top_k=10):
    image_query_cca = cca.transform([image_query_embedding])[0]
    nearest_neighbors = text_tree.nearest_neighbor(image_query_cca, k=top_k, include_distance=False)
    return nearest_neighbors
