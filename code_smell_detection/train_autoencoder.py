import os.path
from autoencoder import train_autoencoder
from io_util import get_all_data


train_validate_ratio = 0.7
data_path = 'data/tokenized_samples'
max_encoding_dim = 1024
smell = 'InsufficientModularization'
out_folder = 'data/results'
layers = [1, 2]
epochs = 5


if not os.path.exists(out_folder):
    os.makedirs(out_folder)

input_data = get_all_data(data_path, smell, train_validate_ratio)
train_autoencoder(max_encoding_dim, input_data, out_folder, smell, layers, epochs)