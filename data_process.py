import os
import pickle
import re
import string
import configparser
import sys

import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim import utils
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer


def clean_text(text):
    # replace special characters with spaces
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # convert to lowercase
    text = text.lower().split()
    # ingore the stopwords
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    # Lemmatization
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(w) for w in text]
    text = " ".join(text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def get_dataframe(root_path):
    # get_data
    # fake_news, label:1, 0 (1: unreliable; 0: reliable)
    if not os.path.isfile(os.path.join(root_path, 'dataset/dataframe.h5')):
        fake_news_df = pd.read_csv(os.path.join(root_path, 'dataset/fake-news/train.csv'), usecols=['label', 'text'])

        # liar, pre_label: 'false', 'half-true', 'mostly-true', 'true', 'barely-true', 'pants-fire'
        liar_df = pd.read_csv(os.path.join(root_path, 'dataset/liar_dataset/train.tsv'), sep='\t', usecols=[1, 2], names=['pre_label', 'text'])
        # assume 'mostly-true', 'true' at 0 means reliable, else 1 means unreliable
        liar_df['label'] = liar_df['pre_label'].map(lambda a: 0 if a in ['mostly-true', 'true'] else 1)

        # concat two dataset
        df = pd.concat([liar_df, fake_news_df])
        df = df[['text', 'label']]

        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
        df.dropna(subset=['text'], inplace=True)
        df['text'] = df['text'].apply(lambda _s: clean_text(_s))
        # reset index
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_hdf(os.path.join(root_path, 'dataset/dataframe.h5'), key='df', mode='w')
    else:
        df = pd.read_hdf(os.path.join(root_path, 'dataset/dataframe.h5'), 'df')

    return df


def get_embeddings_doc2vec(root_path):
    df = get_dataframe()

    x = [TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]) for index, row in
         df['text'].iteritems()]
    y = df['label'].values
    if not os.path.isfile(os.path.join(root_path, 'word_representation/doc2vec_model.pkl')):
        text_model = Doc2Vec(min_count=1, window=5, vector_size=300, sample=1e-4, negative=5, workers=7,
                             epochs=5,
                             seed=1)
        text_model.build_vocab(x)
        text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.epochs)
        text_model.save(os.path.join(root_path, 'word_representation/doc2vec_model.pkl'))
    else:
        text_model = Doc2Vec.load(os.path.join(root_path, 'word_representation/doc2vec_model.pkl'))

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    train_data = np.zeros((train_size, 300))
    test_data = np.zeros((test_size, 300))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        train_data[i] = text_model.dv['Text_' + str(i)]
        train_labels[i] = y[i]

    for i in range(train_size, train_size + test_size):
        test_data[i - train_size] = text_model.dv['Text_' + str(i)]
        test_labels[i - train_size] = y[i]

    return train_data, test_data, train_labels, test_labels


def build_sentence_vector_word2vec(sentence, size, w2v_model):
    sen_vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v_model.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


def get_embeddings_word2vec(root_path):
    df = get_dataframe()

    x = [row.split() for index, row in df['text'].iteritems()]
    y = df['label'].values
    if not os.path.isfile(os.path.join(root_path, 'word_representation/word2vec_model.pkl')):
        text_model = Word2Vec(min_count=1, window=5, vector_size=300, sample=1e-4, negative=5, workers=7,
                              epochs=5,
                              seed=1)
        text_model.build_vocab(x)
        text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.epochs)
        text_model.save(os.path.join(root_path, 'word_representation/word2vec_model.pkl'))
    else:
        text_model = Word2Vec.load(os.path.join(root_path, 'word_representation/word2vec_model.pkl'))

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    train_data = np.zeros((train_size, 300))
    test_data = np.zeros((test_size, 300))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        train_data[i] = build_sentence_vector_word2vec(x[i], 300, text_model)
        train_labels[i] = y[i]

    for i in range(train_size, train_size + test_size):
        test_data[i - train_size] = build_sentence_vector_word2vec(x[i], 300, text_model)
        test_labels[i - train_size] = y[i]

    return train_data, test_data, train_labels, test_labels


def get_embeddings_tfidf(root_path):
    df = get_dataframe(root_path)
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    x_train = df['text'][:train_size]
    x_test = df['text'][train_size:]
    train_labels = df['label'][:train_size].to_numpy()
    test_labels = df['label'][train_size:].to_numpy()

    feature_path = os.path.join(root_path, 'word_representation/tf_idf_feature.pkl')
    tfidftransformer_path = os.path.join(root_path, 'word_representation/tfidftransformer.pkl')

    if not os.path.isfile(feature_path) or not os.path.isfile(tfidftransformer_path):
        vectorizer = CountVectorizer(decode_error="replace")
        tfidftransformer = TfidfTransformer()
        vec_train = vectorizer.fit_transform(x_train)
        train_data = tfidftransformer.fit_transform(vec_train)
        test_data = tfidftransformer.transform(vectorizer.transform(x_test))

        with open(feature_path, 'wb') as fw:
            pickle.dump(vectorizer.vocabulary_, fw)
        with open(tfidftransformer_path, 'wb') as fw:
            pickle.dump(tfidftransformer, fw)

    else:
        vectorizer = CountVectorizer(decode_error="replace",
                                     vocabulary=pickle.load(open(feature_path, "rb")))
        tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))

        train_data = tfidftransformer.transform(vectorizer.transform(x_train))
        test_data = tfidftransformer.transform(vectorizer.transform(x_test))
    return train_data, test_data, train_labels, test_labels


if __name__ == "__main__":

    config_path = os.path.join(os.path.split(os.path.realpath(sys.argv[0]))[0], 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    root_path = config['config']['ROOT_PATH']

    if not os.path.isfile(os.path.join(root_path, 'x_train_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'x_test_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_train_doc2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_test_doc2vec.npy')):
        x_train_doc2vec, x_test_doc2vec, y_train_doc2vec, y_test_doc2vec = get_embeddings_doc2vec(root_path)
        np.save(os.path.join(root_path, 'x_train_doc2vec.npy'), x_train_doc2vec)
        np.save(os.path.join(root_path, 'x_test_doc2vec.npy'), x_test_doc2vec)
        np.save(os.path.join(root_path, 'y_train_doc2vec.npy'), y_train_doc2vec)
        np.save(os.path.join(root_path, 'y_test_doc2vec.npy'), y_test_doc2vec)

    if not os.path.isfile(os.path.join(root_path, 'x_train_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'x_test_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_train_word2vec.npy')) or not os.path.isfile(
            os.path.join(root_path, 'y_test_word2vec.npy')):
        x_train_word2vec, x_test_word2vec, y_train_word2vec, y_test_word2vec = get_embeddings_word2vec(root_path)
        np.save(os.path.join(root_path, 'x_train_word2vec.npy'), x_train_word2vec)
        np.save(os.path.join(root_path, 'x_test_word2vec.npy'), x_test_word2vec)
        np.save(os.path.join(root_path, 'y_train_word2vec.npy'), y_train_word2vec)
        np.save(os.path.join(root_path, 'y_test_word2vec.npy'), y_test_word2vec)

    get_embeddings_tfidf(root_path)
