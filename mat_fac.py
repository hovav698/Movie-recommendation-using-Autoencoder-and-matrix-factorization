import numpy as np
from utils import extract_data, get_data, get_new_ids, get_dicts
import matplotlib.pyplot as plt


# calculate the mse loss according to the matrix factorization formula
def get_loss(m2u):
    # d: movie_id -> (user_ids, ratings)
    N = 0.
    sse = 0
    for j, (u_ids, r) in m2u.items():
        p = W[u_ids].dot(U[:, j]) + B[u_ids] + C[j] + mu[u_ids]
        delta = p - r
        sse += delta.dot(delta)
        N += len(r)
    return sse / N


# the training loop. It implements the expectation maximization for matrix factorization. see paper
def train():
    epochs = 5
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        for i in range(N):
            user_i_movies = train_user2movies[i]
            U_filtered = U[:, user_i_movies]

            unit_matrix = np.eye(K) * lamda
            mat_mul = U_filtered.dot(U_filtered.T)
            user_i_ratings = np.array([train_usermovie2rating[(i, movie_j)] for movie_j in user_i_movies])

            mu[i] = np.mean(user_i_ratings)
            a = mat_mul + unit_matrix

            r_mul_u = U_filtered.dot(user_i_ratings)
            b_mul_u = np.sum(B[i] * U_filtered, axis=1)
            c_mul_u = U_filtered.dot(C[user_i_movies])
            mu_mul_u = np.sum(mu[i] * U_filtered, axis=1)
            b = r_mul_u - b_mul_u - c_mul_u - mu_mul_u

            W[i] = np.linalg.solve(a, b)

            user_i_movies_num = len(user_i_movies)

            r_sum = user_i_ratings.sum()
            w_u_matmul = W[i].dot(U_filtered).sum()
            c_sum = C[user_i_movies].sum()
            mu_sum = mu[i] * user_i_movies_num
            b_sum = r_sum - w_u_matmul - c_sum - mu_sum
            B[i] = b_sum / (user_i_movies_num + lamda)

        for j in range(M):
            movie_j_users = train_movie2users[j]
            w_filtered = W[movie_j_users]
            unit_matrix = np.eye(K) * lamda
            mat_mul = w_filtered.T.dot(w_filtered)
            movie_j_ratings = np.array([train_usermovie2rating[(i, j)] for i in movie_j_users])
            movie_j_mean = np.mean(movie_j_ratings)
            a = mat_mul + unit_matrix
            r_mul_w = movie_j_ratings.dot(w_filtered)
            b_mul_w = B[movie_j_users].dot(w_filtered)
            c_mul_w = np.sum(C[j] * w_filtered, axis=0)
            mu_mul_u = np.sum(movie_j_mean * w_filtered, axis=0)
            b = r_mul_w - b_mul_w - c_mul_w - mu_mul_u

            U[:, j] = np.linalg.solve(a, b)

            movie_j_users_num = len(movie_j_users)

            r_sum = movie_j_ratings.sum()
            w_u_matmul = w_filtered.dot(U[:, j]).sum()
            mu_sum = movie_j_mean * movie_j_users_num
            b_sum = B[movie_j_users].sum()

            c_sum = r_sum - w_u_matmul - b_sum - mu_sum

            C[j] = c_sum / (movie_j_users_num + lamda)

        train_loss = get_loss(train_movie2users_rating)
        test_loss = get_loss(test_movie2users_rating)
        print(f'Epoch {epoch} train loss: {train_loss:.4f} , test loss: {test_loss:.4f}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses


# predicts the rating of all the movies that the user haven't watched
def predict_ratings(original_id):
    movies_idx = np.arange(0, M)
    user_id = user2idx[original_id]
    user_i_movies = user2movies[user_id]
    no_rating_movies = movies_idx[np.isin(movies_idx, user_i_movies) != 1]
    predictions = W[user_id].dot(U[:, no_rating_movies]) + B[user_id] + C[no_rating_movies] + mu[user_id]
    new_movie_idx = [idx2movie[idx.item()] for idx in no_rating_movies]

    return new_movie_idx, predictions


if __name__ == '__main__':
    extract_data()  # Extract the downloaded zip file

    # Limit the data to 10000 users and 2000 movies
    N = 10000
    M = 2000

    # get the shrinked dataset and the corresponding user and movies id
    df,train_df, test_df, users_id, movies_id = get_data(N, M)

    # for convenience, we will change the original id of the users and movies
    # the new ids will be sorted in numerical order
    user2idx, movie2idx ,idx2user,idx2movie= get_new_ids(users_id, movies_id)

    # we don't want to create array of users and movies because it will be sparsed and take too
    # much space. Oneway to overcome this problem is to creates dictionaries, and this is what
    # we
    user2movies, usermovie2rating, movie2users, movie2users_rating = get_dicts(df, users_id, movies_id,
                                                                               user2idx, movie2idx)

    # seperate dictionaries for train and test set
    train_user2movies, train_usermovie2rating, train_movie2users, train_movie2users_rating = get_dicts(train_df,
                                                                                                       users_id,
                                                                                                       movies_id,
                                                                                                       user2idx,
                                                                                                       movie2idx)
    _, _, test_movie2users, test_movie2users_rating = get_dicts(test_df, users_id, movies_id,
                                                                user2idx, movie2idx)

    # The relevant values, array and matrix for the matrix factorization calculation
    K = 10
    lamda = 20
    W = np.random.randn(N, K)
    U = np.random.randn(K, M)
    B = np.zeros(N)
    C = np.zeros(M)
    mu = np.zeros(N)

    # start the train loop using matrix factorization algorithm
    train_losses, test_losses = train()

    # plot the train and test epoch losses
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.show()

    # show prediction for random user
    random_idx = np.random.randint(0, len(users_id))
    prediction = predict_ratings(users_id[random_idx])
    for mid, r in zip(prediction[0], prediction[1]):
        print(f'Movie ID {mid} predicted rating is: {r:.4f}')
