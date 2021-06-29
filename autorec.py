import numpy as np
from utils import extract_data, get_data, get_new_ids, get_dicts
from model import AutoEncoder
from scipy.sparse import lil_matrix
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# converts the tata structure from dictionary to a sparse matrix
def get_sparse_matrix():
    r_matrix = lil_matrix((N, M))
    r_matrix_train = lil_matrix((N, M))
    r_matrix_test = lil_matrix((N, M))

    for i in range(N):
        user_i_movies = user2movies[i]
        np.random.shuffle(user_i_movies)
        split = int(0.8 * len(user_i_movies))

        user_i_train_movies = user_i_movies[:split]
        user_i_test_movies = user_i_movies[split:]

        for j in user_i_train_movies:
            rating_ij = usermovie2rating[(i, j)]
            r_matrix[i, j] = rating_ij
            r_matrix_train[i, j] = rating_ij

        for j in user_i_test_movies:
            rating_ij = usermovie2rating[(i, j)]
            r_matrix[i, j] = rating_ij
            r_matrix_test[i, j] = rating_ij

    return r_matrix, r_matrix_train, r_matrix_test


def calc_loss(pred, target):
    non_zero_idx = target.nonzero(as_tuple=True)
    loss = torch.mean((pred[non_zero_idx] - target[non_zero_idx]) ** 2)
    return loss


def train():
    epochs = 100
    batch_size = 64
    n_batchs = N // batch_size
    train_losses = []
    test_losses = []

    for epoch in range(epochs):

        # np.random.shuffle(index)
        # r_matrix=r_matrix[index, :]
        train_epoch_losses = []
        test_epoch_losses = []

        for n in range(n_batchs):
            model.train()

            train_ratings_batch = torch.tensor(r_matrix_train[n * batch_size:(n + 1) * batch_size].toarray(),
                                               dtype=torch.float32).to(device)

            optimizer.zero_grad()
            train_preds_batch = model(train_ratings_batch)
            train_loss = calc_loss(train_preds_batch, train_ratings_batch)
            train_loss.backward()
            train_epoch_losses.append(train_loss.item())
            optimizer.step()

            model.eval()

            test_ratings_batch = torch.tensor(r_matrix_test[n * batch_size:(n + 1) * batch_size].toarray(),
                                              dtype=torch.float32).to(device)
            test_preds_batch = model(test_ratings_batch)
            test_loss = calc_loss(test_preds_batch, test_ratings_batch)
            test_epoch_losses.append(test_loss.item())

        train_avg_epoch_loss = np.mean(train_epoch_losses)
        train_losses.append(train_avg_epoch_loss)

        test_avg_epoch_loss = np.mean(test_epoch_losses)
        test_losses.append(test_avg_epoch_loss)

        print(f'Epoch {epoch} train loss: {train_avg_epoch_loss:.4f}   ,   test loss: {test_avg_epoch_loss:.4f}')

    return train_losses, test_losses


# predicts the rating of all the movies that the user haven't watched
def predict_ratings(original_id):
    user_id = user2idx[original_id]
    movie_vector = torch.tensor(r_matrix[0].toarray(), dtype=torch.float32).to(device)
    user_preds = model(movie_vector)
    zero_idx = torch.nonzero(movie_vector == 0, as_tuple=True)
    new_movies_prediction = user_preds[zero_idx]
    new_movie_idx = [idx2movie[idx.item()] for idx in zero_idx[1]]
    return new_movie_idx, new_movies_prediction


if __name__ == "__main__":
    extract_data()  # Extract the downloaded zip file

    # Limit the data to 10000 users and 2000 movies
    N = 10000
    M = 2000

    # get the shrinked dataset and the corresponding user and movies id
    df, train_df, test_df, users_id, movies_id = get_data(N, M)

    # for convenience, we will change the original id of the users and movies
    # the new ids will be sorted in numerical order
    user2idx, movie2idx, idx2user, idx2movie = get_new_ids(users_id, movies_id)

    # we will use the dictionaries that were created for the matrix factorization problem
    # we will only use it for creating the sparse matrix
    user2movies, usermovie2rating, movie2users, movie2users_rating = get_dicts(df, users_id, movies_id,
                                                                               user2idx, movie2idx)
    # create the sparse matrix
    r_matrix, r_matrix_train, r_matrix_test = get_sparse_matrix()

    K = 700

    model = AutoEncoder(M, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = train()

    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.show()

    prediction = predict_ratings(users_id[100])
    for mid, r in zip(prediction[0], prediction[1]):
        print(f'Movie ID {mid} predicted rating is: {r:.4f}')
