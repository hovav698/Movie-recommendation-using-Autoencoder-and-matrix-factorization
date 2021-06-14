This repository contains pytorch implementation of two Recommender System algorithms. I will first briefly go through the theory and explain the basic implementation.

**Dataset description, and what is the goal**

The Dataset consist of a very large amount of users (200K+) and and movies (40K+), and the corresponding rating of the user to the movie. The goal is to build a model that can predict what is the rating of movies that the user haven't seen yet. by doing that we can recommended the users movies that there is a big chance they will like.

**The algorithms**

I will introduce 2 algorithms for solving the problem. The first algorithm is matrix factorization. The second algorithm called Autorec - a recommender system using Auto-Encoder.

**Data Preperation**

The original data is divided into very long list of tuples - each tuple consist of user ID, movie ID and the corresponding rating. For using the algorithm I will need to convert it to a different data structure that will allow to get the movie rating directly from the userID and movieID indexes. I won't create a normal 2D matrix of the ratings because it will take too much space and almost all the rating values will be empty, since each user haven't rated most of the movies. For the Matrix Factorization algorithm I will conver the data and put it into dictionaries. That data structure will ease the query lookup. For the Autorec alogorithm I will use the spicy sparse matrix - it stores the a sparse matrices in efficient way and it will allow to use 2D matrix for representing the data.

**Matrix Factorization**

The basic behind matrix factorization is trying to reconstract the original movie rating matrix ![image](https://user-images.githubusercontent.com/71300410/121802175-de563980-cc43-11eb-92ce-2f073f8aa8d9.png) from smaller matrices![image](https://user-images.githubusercontent.com/71300410/121802098-91726300-cc43-11eb-8a18-459647f953c7.png) and ![image](https://user-images.githubusercontent.com/71300410/121802142-cc749680-cc43-11eb-9237-6a73b7988587.png), in a way that the product ![image](https://user-images.githubusercontent.com/71300410/121802232-09408d80-cc44-11eb-96da-58193a09f824.png) will be close as possible to the original matrix A. 

<img width="500" alt="matfac" src="https://user-images.githubusercontent.com/71300410/121802495-43f6f580-cc45-11eb-98f8-ab646dddd519.PNG">

The loss function will be defined as the squqare root distance between the original matrix and the decomposited matrix. I will use the expectation maximization algorithm for finding the relevant values for the rating prediction. All the math is explained in the following [paper](https://papers.nips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf).

**AutoRec Algorithm**

The autorec algorithm is a simple autoenconder. The model input is a vector from the 2D sparse matrix of user and movies rating - a vector of movies and it's corresponding rating.  The model output is a vector of the same dimentions as the output. The loss function will be the MSE loss between the output and the inputs - after the training stage the model will learn the best way to recreate the input from the model hidden layer that functions as a bottleneck.


**Result**

The result of the two models were very close, however the training time of the autoncoder model was much faster.







