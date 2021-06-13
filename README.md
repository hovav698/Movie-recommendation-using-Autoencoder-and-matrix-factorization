This repository contains pytorch implementation of two Recommender System algorithms. I will first briefly go through the theory and explain the basic implementation.

**Dataset description, and what is the goal**

The Dataset consist of a very large amount of users (200K+) and and movies (40K+), and it's corresponding rating of the user to the the movie. The goal is to build a model that can predict what is the rating of movies that the user haven't seen yet. by doing that we can recommended the users movies that there is a big chance they will like.

**The algorithms**

I will implement 2 algorithms for solving the problem. The first algorithm is matrix factorization. It described in details on the following [tutorial](https://developers.google.com/machine-learning/recommendation/collaborative/matrix). The second algorithm is algorithm called Autorec - a recommender system using Auto-Encoder.

**Data Preperation**

The original data is divided into very long list of tuples - each tuple consist of user ID, movie ID and the corresponding rating. For using the algorithm I will need to convert it to a different data structure that will allow me to get the movie rating directly from the userID and movieID indexes. I won't create a normal 2D matrix of the ratings because it will take too much space and almost all the rating values will be empty, since the each user haven't rated most of the movies. For the Matrix Factorization algorithm I will conver the data and put it into dictionaries that will ease the query lookup. For the autorec alogorithm I will use the spicy sparse matrix - it stores the a sparse matrices in efficient way and it will allow me to use 2D matrix for representing the data.

**Matrix Factorization**
