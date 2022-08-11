import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data_process import get_embeddings_doc2vec, get_embeddings_word2vec, get_embeddings_tfidf
import joblib


def knn_doc2vec(root_path):
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

    knn = KNeighborsClassifier()
    knn.fit(x_train_doc2vec, y_train_doc2vec)
    y_predict = knn.predict(x_test_doc2vec)
    m = y_test_doc2vec.shape[0]
    n = (y_test_doc2vec != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")

    joblib.dump(knn, 'knn_doc2vec.pkl')


def knn_word2vec(root_path):
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

    knn = KNeighborsClassifier()
    knn.fit(x_train_word2vec, y_train_word2vec)
    y_predict = knn.predict(x_test_word2vec)
    m = y_test_word2vec.shape[0]
    n = (y_test_word2vec != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")

    joblib.dump(knn, 'knn_word2vec.pkl')


def knn_tf_idf(root_path):
    # Accuracy = 83.18%
    x_train_tf_idf, x_test_tf_idf, y_train_tf_idf, y_test_tf_idf = get_embeddings_tfidf(root_path)
    knn = KNeighborsClassifier()
    knn.fit(x_train_tf_idf, y_train_tf_idf)
    y_predict = knn.predict(x_test_tf_idf)
    m = y_test_tf_idf.shape[0]
    n = (y_test_tf_idf != y_predict).sum()
    print("Accuracy = " + format((m - n) / m * 100, '.2f') + "%")
    joblib.dump(knn, 'knn_tfidf.pkl')


if __name__ == "__main__":
    import configparser, sys

    config_path = os.path.join(os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0], 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    root_path = config['config']['ROOT_PATH']
    knn_doc2vec(root_path)
    knn_word2vec(root_path)
    knn_tf_idf(root_path)
