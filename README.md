# svdRec - A Recommender System

This is a Python 3 based recommendation system based on Singular Value
Decomposition (SVD). The module allows users to load from a CSV or from a numpy
array/matrix. This is converted to a sparse matrix, and SVD is computed to
convert the user/item matrix into a k-dimensional space where recommendations
are computed based on vector overlaps. 

With this system you can pull out recommendations for users, find similar
items, find users with similar history, and make recommendations based on user
similarity.

An example of using this system can be found here: [Movie Lens Recommender](svdRec_example.ipynb).
