import os
import nltk
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews

# b = Word2Vec(brown.sents())
# mr = Word2Vec(movie_reviews.sents())
# t = Word2Vec(treebank.sents())

if __name__ == '__main__':
    # data = [['teamsar', 'muliadi', 'hobi', 'makan', 'nasi', 'remes'],
    #         ['angga', 'sanjaya', 'lingga', 'makan', 'tahu', 'isi'],
    #         ['disana', 'ada', 'jual', 'makanan', 'ringan', 'enak'], ['tidak', 'enak', 'kalau', 'masak', 'sendiri'],
    #         ['kutunggu', 'makanan', 'bergizi']]
    data = []
    for root, dirs, files in os.walk("..\SNS\political_corpus\\"):
        for file in files:
            if not file.endswith(".py") and not file.endswith(".txt"):
                f = open(os.path.join(root, file), 'r', encoding='utf8')
                lines = f.readlines()
                for line in lines:
                    tokens = nltk.word_tokenize(line)
                    data.append(tokens)
                # print(os.path.join(root, file))

    wv = Word2Vec(sentences=data, min_count=2)
    print(wv['pencurian'], wv.most_similar('pencurian'))
    # print(wv['makan'], wv.most_similar('makan'))
