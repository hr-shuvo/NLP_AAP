import io
import torch

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn import metrics

import config
import dataset
import engine
import lstm


def load_vectors(fname):
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]

    return embedding_matrix


def run(df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)

    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print('Fitting tokenizer')

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)

    train_dataset = dataset.IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    valid_dataset = dataset.IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    print(f'{fold}: Loading embedding...')

    embedding_dict = load_vectors('data/crawl-300d-2M.vec')
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device: {}'.format(device))

    model = lstm.LSTM(embedding_matrix)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('Training model...')

    best_accuracy = 0
    early_stopping_counter = 0
    for epoch in range(config.EPOCHS):
        engine.train(train_data_loader, model, optimizer, device)

        outputs, targets = engine.evaluate(valid_data_loader, model, device)

        outputs = np.array(outputs) >= 0.5

        accuracy = metrics.accuracy_score(targets, outputs)

        print(f'Fold: {fold}, Epoch: {epoch}, Accuracy: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

            if early_stopping_counter > 2:
                break


if __name__ == '__main__':
    df = pd.read_csv('data/imdb_folds.csv')

    print(df.head())

    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)
    run(df, fold=5)



