from learning.classifier.classifier import Classifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif, chi2
import simplemma
import numpy as np
import gensim

wordnet_lemmatizer = WordNetLemmatizer()

class Embedding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def calculate_fasttext_mean_vector(self, model, words):
        words = [word for word in words if word in model.vocab]
        if len(words) >= 1:
            return np.mean(model[words], axis=0)
        else:
            return []

    def transform(self, X):
        fasttext_model = gensim.models.KeyedVectors.load_word2vec_format("data/fasttext/wiki.pt.vec")
        new = []
        for sentence in X:
            token_words = word_tokenize(sentence)
            emb_sentence= self.calculate_fasttext_mean_vector(fasttext_model, token_words)
            new.append(emb_sentence)
        return new

class Lemmarize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new = []
        for sentence in X:
            token_words = word_tokenize(sentence)
            stem_sentence = []
            for word in token_words:
                #stem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
                stem_sentence.append(simplemma.lemmatize(word, lang='pt'))
            new.append(" ".join(stem_sentence))
        return new

class POS(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        spacy.prefer_gpu()
        nlp = spacy.load("pt_core_news_lg")
        new = []
        for sentence in X:
            doc = nlp(sentence)
            json_doc = doc.to_json()
            token = json_doc['tokens']
            tag = []
            for t in token:
                tag.append(t['tag'])
            new.append(" ".join(tag))
        return new

class SVMClassifier(Classifier):
    def __init__(self, model_file_path=None):
        super(SVMClassifier, self).__init__(model_file_path)
        self.pipeline = Pipeline([
                                  ('lemma', Lemmarize()),
                                  #('tf-idf', TfidfVectorizer(max_df=0.9, min_df=3, max_features=2000, ngram_range=(1,4))),
                                  ('fasttext', Embedding()),
                                  ('svm', SVC())
                                 ]
                                )
        self.parameters = {'svm__C': [0.1,1, 10, 100],
                           'svm__gamma': [1,0.1,0.01,0.001],
                           'svm__kernel': ['rbf', 'poly', 'sigmoid']
                          }
