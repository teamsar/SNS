import os
import nltk
import pickle
from gensim.models import Word2Vec


class IndonesianWord2Vec:
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, retrain=False):
        self.sentences = sentences
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.retrain = retrain

    def retrain_word2vec(self):
        if self.retrain:
            data = []
            for root, dirs, files in os.walk('..\\SNS\\political_corpus\\'):
                for file in files:
                    if not file.endswith(".py") and not file.endswith(".txt"):
                        f = open(os.path.join(root, file), 'r', encoding='utf8')
                        lines = f.readlines()
                        for line in lines:
                            tokens = nltk.word_tokenize(line)
                            data.append(tokens)
            wv = Word2Vec(sentences=data, min_count=self.min_count)
            with open('..\\SNS\\saved_model\\indonesian_word2vec_saved_model.pickle', 'wb') as pickle_file:
                pickle.dump(wv, pickle_file)
            return wv
        else:
            with open('..\\SNS\\saved_model\\indonesian_word2vec_saved_model.pickle', 'rb') as pickle_file:
                return pickle.load(pickle_file)
