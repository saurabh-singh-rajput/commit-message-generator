import numpy as np
import pandas as pd
from tensorflow.python.keras import regularizers
# import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from io_util import write_result, get_out_file


# def _get_plot(history):
#     plt.plot(history['loss'])
#     plt.plot(history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper right')
#     plt.show()


def compute_metrics(conf_matrix):
    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    f1 = (2 * precision * recall) / (precision + recall)
    print("precision: " + str(precision) + ", recall: " + str(recall) + ", f1: " + str(f1))
    return precision, recall, f1


def find_optimal(error_df):
    optimal_threshold = 1000
    max_f1 = 0
    max_pr = 0
    max_re = 0
    for threshold in range(1000, 400000, 5000):
        print("Threshold: " + str(threshold))
        y_pred = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.True_class, y_pred)
        precision, recall, f1 = compute_metrics(conf_matrix)
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold
            max_pr = precision
            max_re = recall
    print(f"Result optimal_threshold={optimal_threshold}, max_precision={max_pr}, max_recall={max_re}, max_f1={max_f1}")
    return optimal_threshold, max_pr.numpy(), max_re.numpy(), max_f1.numpy()


def autoencoder_dense(data, layers=1, encoding_dimension=32, epochs=10, with_bottleneck=True):
    encoding_dim = encoding_dimension
    input_layer = Input(shape=(data.max_input_length,))
    no_of_layers = layers
    prev_layer = input_layer
    print(f"Training Autoencoder with encoding:{encoding_dim}, layers={layers}")
    for i in range(no_of_layers):
        encoder = Dense(int(encoding_dim / pow(2, i)),
                        activation="relu",
                        activity_regularizer=regularizers.l1(10e-3))(prev_layer)
        prev_layer = encoder
    # bottleneck
    if with_bottleneck:
        prev_layer = Dense(int(encoding_dim / pow(2, no_of_layers)),
                           activation="relu")(prev_layer)
    for j in range(no_of_layers - 1, -1, -1):
        decoder = Dense(int(encoding_dim / pow(2, j)),
                        activation='relu')(prev_layer)
        prev_layer = decoder
    prev_layer = Dense(data.max_input_length,
                       activation='relu')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()

    batch_sizes = [32, 64, 128]
    # batch_sizes = [32, 64, 128, 256, 512]
    b_size = int(len(data.train_data) / batch_sizes[len(batch_sizes) - 1])
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1

    val_split = 0.2
    history = autoencoder.fit(data.train_data,
                              data.train_data,
                              epochs=epochs,
                              batch_size=batch_sizes[b_size],
                              verbose=1,
                              validation_split=val_split,
                              shuffle=True).history

    # _get_plot(history)

    predictions = autoencoder.predict(data.eval_data)
    mse = np.mean(np.power(data.eval_data - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': data.eval_labels})

    optimal_threshold, max_pr, max_re, max_f1 = find_optimal(error_df)
    return optimal_threshold, max_pr, max_re, max_f1


def train_autoencoder(max_encoding_dim, input_data, out_folder, smell, layers, epochs):
    max_encoding_dimension = min(max_encoding_dim, input_data.max_input_length)
    encoding_dim = [int(max_encoding_dimension / 4), int(max_encoding_dimension / 2), int(max_encoding_dimension)]

    outfile = get_out_file(out_folder=out_folder, smell=smell)
    write_result(outfile, "Encoding_dim,threshold,epoch,bottleneck,layer,precision,recall,f1\n")
    for layer in layers:
        for bottleneck in [True]:
            for encoding in encoding_dim:
                try:
                    optimal_threshold, max_pr, max_re, max_f1 = autoencoder_dense(input_data, layers=layer,
                                                                                  epochs=epochs,
                                                                                  encoding_dimension=encoding,
                                                                                  with_bottleneck=bottleneck)
                except ValueError as error:
                    print(error)
                    optimal_threshold = -1
                    max_pr = -1
                    max_re = -1
                    max_f1 = -1
                write_result(outfile, f"{encoding},{optimal_threshold},{epochs},{bottleneck},{layer},{max_pr},{max_re},{max_f1}\n")
                             # str(encoding) + "," + str(optimal_threshold) + "," + str(
                             #     epochs) + "," + str(bottleneck) + "," + str(layer) + "," +
                             # str(max_pr.numpy()) + "," + str(max_re.numpy()) + "," + str(max_f1.numpy()) + "\n")

