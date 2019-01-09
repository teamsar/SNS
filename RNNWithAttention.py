import os
import nltk
import pickle
import keras
import numpy as np
import tensorflow as tf
from IndonesianWord2Vec import IndonesianWord2Vec
from numpy import array_equal
from attention_decoder import AttentionDecoder
from keras.models import Sequential
from keras.layers import LSTM


# prepare data for the LSTM
def get_pair(n_in, n_out, tokens):
    sequence_in = tokens
    sequence_out = np.concatenate((sequence_in[:n_out],
                                   [[0 for _ in range(100)] for _ in range(n_in - n_out)]), axis=0)

    X = sequence_in
    y = sequence_out

    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


def get_train_data(min_token_in_sentence=10):
    if not os.path.exists('..\\SNS\\saved_model\\trained_data_with_pad.pickle'):
        data = []
        for root, dirs, files in os.walk('..\\SNS\\political_corpus\\'):
            for file in files:
                if not file.endswith(".py") and not file.endswith(".txt"):
                    f = open(os.path.join(root, file), 'r', encoding='utf8')
                    lines = f.readlines()
                    for line in lines:
                        tokens = nltk.word_tokenize(line)  # list of tokens
                        len_token = len(tokens)
                        if len_token == min_token_in_sentence:
                            data.append(tokens)
                        else:
                            tmp = add_pad(tokens=tokens,
                                          max_pads=min_token_in_sentence)
                            data.extend(tmp)
        with open('..\\SNS\\saved_model\\trained_data_with_pad.pickle', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
    else:
        with open('..\\SNS\\saved_model\\trained_data_with_pad.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)
    return data


def get_test_data(min_token_in_sentence=10):
    data = list()
    f = open('..\\SNS\\test_data\\test1', 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        tokens = nltk.word_tokenize(line)  # list of tokens
        len_token = len(tokens)
        if len_token == min_token_in_sentence:
            data.append(tokens)
        else:
            tmp = add_pad(tokens=tokens,
                          max_pads=min_token_in_sentence)
            data.extend(tmp)
    return data


def add_pad(tokens, max_pads, data=None):
    len_token = len(tokens)
    if len_token > max_pads:
        tmp = tokens[:max_pads]
        if data is None: data = []
        data.append(tmp)
        tokens = tokens[max_pads:]
        add_pad(tokens, max_pads, data)
    else:
        if data is None: data = []
        tokens.extend(['<PAD>' for _ in range(max_pads - len_token)])
        data.append(tokens)
    return data


def get_word_vector(wv, data, size_output_vector_per_token):
    words_vectors = list()
    for tokens in data:
        tmp = list()
        for token in tokens:
            if '<PAD>' in token:
                tmp.append(np.zeros((size_output_vector_per_token,)))
            else:
                try:
                    vector = wv[token]
                    tmp.append(vector)
                except:
                    tmp.append(np.zeros((size_output_vector_per_token,)))
        words_vectors.append(np.array(tmp))
    return np.array(words_vectors)


if __name__ == '__main__':
    # configure problem
    size_output_vector_per_token = 100
    min_token_in_sentence = 5
    n_features = size_output_vector_per_token
    n_timesteps_in = min_token_in_sentence
    n_timesteps_out = 2

    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    wv = IndonesianWord2Vec(retrain=True, min_count=3,
                            size=size_output_vector_per_token).retrain_word2vec()

    data = get_train_data(min_token_in_sentence=min_token_in_sentence)
    data = get_word_vector(wv=wv, data=data, size_output_vector_per_token=size_output_vector_per_token)

    with tf.device('/gpu:0'):
        # define model
        model = Sequential()
        model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
        model.add(AttentionDecoder(150, n_features))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        # train LSTM
        for tokens in data:
            # generate new random sequence
            X, y = get_pair(n_timesteps_in, n_timesteps_out, tokens)
            # fit model for one epoch on this sequence
            model.fit(X, y, epochs=1, verbose=2)

        # evaluate LSTM
        validation_data = get_test_data(min_token_in_sentence=min_token_in_sentence)
        validation_data = get_word_vector(wv=wv, data=validation_data,
                                          size_output_vector_per_token=size_output_vector_per_token)
        total, correct = len(validation_data), 0
        for tokens in validation_data:
            X, y = get_pair(n_timesteps_in, n_timesteps_out, tokens)
            yhat = model.predict(X, verbose=0)
            if array_equal(y[0], yhat[0]):
                correct += 1
        print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))

        # # spot check some examples
        # for _ in range(10):
        #     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        #     yhat = model.predict(X, verbose=0)
        #     print('Expected:', y[0], 'Predicted', yhat[0])
