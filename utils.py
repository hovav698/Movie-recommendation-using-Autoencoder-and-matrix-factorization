# for downloading the data type
# kaggle datasets download -d rounakbanik/the-movies-dataset in the terminal

import kaggle
import shutil
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle


def extract_data():
    if not os.path.exists('data'):
        shutil.unpack_archive('archive.zip', 'data')


# shrink the dataset and split it to train and test set. return the datsets and the corresponding
# users and movies id.
# The users and movies that were selected are the most popular movies and the users that rated the highest
# number of movies
def get_data(N, M):
    ratings_df = pd.read_csv('data/ratings.csv')
    ratings_df = ratings_df.drop(columns='timestamp')
    movies_id = ratings_df['userId'].unique()
    users_id = ratings_df['movieId'].unique()

    num_of_users = len(movies_id)
    num_of_movies = len(users_id)

    print("Total number of users:", num_of_users)
    print("Total number of movies:", num_of_movies)

    ratings_df = shuffle(ratings_df)

    users_counter = Counter(ratings_df['userId'])
    most_common_users = np.array(users_counter.most_common())[:, 0]

    movies_counter = Counter(ratings_df['movieId'])
    most_common_movies = np.array(movies_counter.most_common())[:, 0]

    users_id = most_common_users[:N]
    movies_id = most_common_movies[:M]

    small_df = ratings_df[ratings_df['userId'].isin(users_id) & ratings_df['movieId'].isin(movies_id)]

    split = int(len(small_df) * 0.8)
    train_df = small_df[:split]
    test_df = small_df[split:]

    return small_df,train_df, test_df, users_id, movies_id


def get_new_ids(users_id, movies_id):
    user2idx = {}
    movie2idx = {}

    idx2user = {}
    idx2movie = {}

    idx = 0
    for user_i in users_id:
        user2idx[user_i] = idx
        idx2user[idx] = user_i
        idx += 1

    idx = 0
    for movie_j in movies_id:
        movie2idx[movie_j] = idx
        idx2movie[idx] = movie_j
        idx += 1

    return user2idx, movie2idx, idx2user, idx2movie


# the function return lists of movies and rating for specific user
def get_user_movies_and_ratings(df, user_id, movies_id):
    movies_and_ratings = df[df['userId'] == user_id]
    movies_indexes = movies_and_ratings['movieId'].isin(movies_id)
    m_id = movies_and_ratings['movieId'][movies_indexes]
    ratings = movies_and_ratings['rating'][movies_indexes]

    return m_id, ratings


# the function return lists of all the users that rated specific movies
def get_movie_users(df, movie_id, users_id):
    users_and_ratings = df[df['movieId'] == movie_id]
    u_id = users_and_ratings['userId'][users_and_ratings['userId'].isin(users_id)]

    return u_id


# the function returns 4 different dictionaries:
# user2movies - the key is the userd id, the value is a list of all the movies that the user has rated
# usermovie2rating - the key is a tuple of user and movie, the value is the corresponding rating
# movie2users_rating - the key is the movie, the values is the users and their rating for that movie
# movie2users - the ket is the movie, the values is a list of all the users that rated this movie
def get_dicts(df, users_id, movies_id, user2idx, movie2idx):
    M = len(movies_id)

    user2movies = {}
    usermovie2rating = {}
    movie2users_rating = {}
    movie2users = {}
    for uid in users_id:
        m_ids, m_ratings = get_user_movies_and_ratings(df, uid, movies_id)
        uid_new = user2idx[uid]
        mids_new = np.array([movie2idx[mid] for mid in m_ids])
        user2movies[uid_new] = mids_new

        for m_id, m_rating in zip(mids_new, m_ratings):
            usermovie2rating[(uid_new, m_id)] = m_rating

    for mid in movies_id:
        mid_new = movie2idx[mid]
        movie_users = get_movie_users(df, mid, users_id)
        movie_users_new = np.array([user2idx[uid] for uid in movie_users])
        movie2users[mid_new] = movie_users_new

    for j in range(M):
        movie_j_users = movie2users[j]
        movie_j_ratings = np.array([usermovie2rating[(i, j)] for i in movie_j_users])
        movie2users_rating[j] = (movie_j_users, movie_j_ratings)

    return user2movies, usermovie2rating, movie2users, movie2users_rating
