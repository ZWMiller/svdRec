# svdRec - A Recommender System

This is a Python 3 based collaborative filtering recommendation system based on Singular Value
Decomposition (SVD). The module allows users to load from a CSV or from a numpy
array/matrix. This is converted to a sparse matrix, and SVD is computed to
convert the user/item matrix into a k-dimensional space where recommendations
are computed based on vector overlaps. 

With this system you can pull out recommendations for users, find similar
items, find users with similar history, and make recommendations based on user
similarity.

An example of using this system can be found here: [Movie Lens
Recommender](svdRec_example.ipynb). In short, given a user item matrix this
module can reliably find similar items based on user input. Below, the ID for
Toy Story 2 is input and the most similar items are _Toy Story_, _A Bug's Life_,
_Who Framed Roger Rabbit?_, and _Finding Nemo_. All of these are animated
children's movies, with the majority being Pixar movies. 

```python
MOVIE_ID = 3114 # Toy Story 2
for item in svd.get_similar_items(MOVIE_ID,show_similarity=True):
      print(item)
      print(svd.get_item_name(item[0]),'\n')
```
```
(3114, 0.12452773823527524)
Toy Story 2 (1999) 

(1, 0.096984857294616089)
Toy Story (1995) 

(2355, 0.043104443630875205)
Bug's Life, A (1998) 

(2987, 0.041949127538017023)
Who Framed Roger Rabbit? (1988) 

(6377, 0.040522854363774369)
Finding Nemo (2003) 
```
