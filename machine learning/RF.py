import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_process import get_embeddings_doc2vec, get_embeddings_word2vec, get_embeddings_tfidf
import joblib


def RF_doc2vec(root_path):
    # Accuracy = 52.49%
    if not os.path.isfile(os.path.join(root_path, 'x_train_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'x_test_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_train_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_test_doc2vec.npy')):
        x_train_doc2vec, x_test_doc2vec, y_train_doc2vec, y_test_doc2vec = get_embeddings_doc2vec(root_path)
        np.save(os.path.join(root_path, 'x_train_doc2vec.npy'), x_train_doc2vec)
        np.save(os.path.join(root_path, 'x_test_doc2vec.npy'), x_test_doc2vec)
        np.save(os.path.join(root_path, 'y_train_doc2vec.npy'), y_train_doc2vec)
        np.save(os.path.join(root_path, 'y_test_doc2vec.npy'), y_test_doc2vec)

    else:
        print("strat")
        x_train_doc2vec = np.load(os.path.join(root_path, 'x_train_doc2vec.npy'))
        x_test_doc2vec = np.load(os.path.join(root_path, 'x_test_doc2vec.npy'))
        y_train_doc2vec = np.load(os.path.join(root_path, 'y_train_doc2vec.npy'))
        y_test_doc2vec = np.load(os.path.join(root_path, 'y_test_doc2vec.npy'))

    forest = RandomForestClassifier(n_estimators=200)
    forest.fit(x_train_doc2vec, y_train_doc2vec)
    y_predict = forest.predict(x_test_doc2vec)
    m = y_test_doc2vec.shape[0]
    n = (y_test_doc2vec != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")

    joblib.dump(forest, 'rf_doc2vec.pkl')


def RF_word2vec(root_path):
    # Accuracy = 82.15%
    if not os.path.isfile(os.path.join(root_path, 'x_train_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'x_test_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_train_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_test_word2vec.npy')):
        x_train_word2vec, x_test_word2vec, y_train_word2vec, y_test_word2vec = get_embeddings_word2vec()
        np.save(os.path.join(root_path, 'x_train_word2vec'), x_train_word2vec)
        np.save(os.path.join(root_path, 'x_test_word2vec'), x_test_word2vec)
        np.save(os.path.join(root_path, 'y_train_word2vec'), y_train_word2vec)
        np.save(os.path.join(root_path, 'y_test_word2vec'), y_test_word2vec)

    else:
        x_train_word2vec = np.load(os.path.join(root_path, 'x_train_word2vec.npy'))
        x_test_word2vec = np.load(os.path.join(root_path, 'x_test_word2vec.npy'))
        y_train_word2vec = np.load(os.path.join(root_path, 'y_train_word2vec.npy'))
        y_test_word2vec = np.load(os.path.join(root_path, 'y_test_word2vec.npy'))

    forest = RandomForestClassifier(n_estimators=200)
    forest.fit(x_train_word2vec, y_train_word2vec)
    y_predict = forest.predict(x_test_word2vec)
    m = y_test_word2vec.shape[0]
    n = (y_test_word2vec != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")

    joblib.dump(forest, 'rf_word2vec.pkl')


def RF_tf_idf():
    # Accuracy = 83.18%
    x_train_tf_idf, x_test_tf_idf, y_train_tf_idf, y_test_tf_idf = get_embeddings_tfidf(root_path)
    forest = RandomForestClassifier(n_estimators=200)
    forest.fit(x_train_tf_idf, y_train_tf_idf)
    y_predict = forest.predict(x_test_tf_idf)
    m = y_test_tf_idf.shape[0]
    n = (y_test_tf_idf != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")
    joblib.dump(forest, 'rf_tfidf.pkl')


if __name__ == "__main__":
    import configparser, sys

    config_path = os.path.join(os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0], 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    root_path = config['config']['ROOT_PATH']

    RF_doc2vec(root_path)
    RF_word2vec(root_path)
    RF_tf_idf(root_path)
    # Accuracy = 52.36 %
    # Accuracy = 82.28 %
    # Accuracy = 83.76 %
