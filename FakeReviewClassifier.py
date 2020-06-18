import csv
import random
import sys

import numpy as np
import tensorflow_hub as hub
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

random.seed(2009)
STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words('english')
import nltk

nltk.download('stopwords')


def remove(word, stop):
    return ''.join(STEMMER.stem(x) + " " for x in word.split() if STEMMER.stem(x) not in stop);


if __name__ == '__main__':

    import logging

    logging.getLogger().setLevel(logging.INFO)

    Y = []
    X = []
    with open('fake_review_dataset.csv') as csv_file:
        for row in csv.reader(csv_file, delimiter=','):
            Y.append(row[0])
            X.append(remove(row[1], STOPWORDS))

    sentence_embeddings = hub.text_embedding_column(
        "fake_review_embedding",
        module_spec="https://tfhub.dev/google/universal-sentence-encoder/2"
    )

    combined_list = list(zip(Y, X))
    random.shuffle(combined_list)
    Y, X = zip(*combined_list)

    tf.autograph.set_verbosity(10)
    train_size = int(len(X) * .8)
    train_descriptions = X[:train_size]
    train_genres = Y[:train_size]

    test_descriptions = X[train_size:]
    test_genres = Y[train_size:]

    encoder = LabelBinarizer()
    encoder.fit_transform(train_genres)
    train_encoded = encoder.transform(train_genres)
    test_encoded = encoder.transform(test_genres)
    num_classes = len(encoder.classes_)
    estimator = tf.estimator.DNNClassifier(
        n_classes=2,
        hidden_units=[1024, 512, 256],
        feature_columns=[sentence_embeddings],
        optimizer=lambda: tf.keras.optimizers.Adam(
            learning_rate=tf.compat.v1.train.exponential_decay(
                learning_rate=0.15,
                global_step=tf.compat.v1.train.get_global_step(),
                decay_steps=10000,
                decay_rate=0.96))
    )
    features = {
        "fake_review_embedding": np.array(train_descriptions)
    }
    labels = np.array(train_encoded)
    features_test = {
        "fake_review_embedding": np.array(test_descriptions)
    }
    labels_test = np.array(test_encoded)


    def input_fn(features, labels, batch_size, repeat, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(1000)
        if repeat:
            dataset = dataset.repeat()
        return dataset.batch(batch_size)


    estimator.train(input_fn=lambda: input_fn(features, labels, 128, True, True),
                    steps=1000)
    y = estimator.predict(
        input_fn=lambda: input_fn(features_test, labels_test, 128, False, False))
    y = list(a['class_ids'] for a in y)
    print(classification_report(labels_test, y), file=sys.stderr)
    print(confusion_matrix(labels_test, y), file=sys.stderr)
    print(f1_score(y, labels_test, average='micro'))
