from distutils.core import setup
setup(
  name = 'svdRec',
  packages = ['svdRec'], # this must be the same as the name above
  version = '0.1.1',
  description = 'A Python3 Collaborative Recommendation Engine with Sparse Matrices',
  author = 'Zachariah Miller',
  author_email = 'zaglamir@gmail.com',
  url = 'https://github.com/ZWMiller/svdRec', # use the URL to the github repo
  download_url = 'https://github.com/zwmiller/svdRec/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['recommender', 'sparse', 'matrix', 'recsys', 'svd'], # arbitrary keywords
  classifiers = [],
  install_requires = ['scipy','numpy'], # required packages
)
