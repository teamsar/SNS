from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank

# b = Word2Vec(brown.sents())
# mr = Word2Vec(movie_reviews.sents())
# t = Word2Vec(treebank.sents())

if __name__ == '__main__':
    data = [['teamsar', 'muliadi', 'hobi', 'makan', 'nasi', 'remes'],
            ['angga', 'sanjaya', 'lingga', 'makan', 'tahu', 'isi'],
            ['disana', 'ada', 'jual', 'makanan', 'ringan', 'enak'], ['tidak', 'enak', 'kalau', 'masak', 'sendiri'],
            ['kutunggu', 'makanan', 'bergizi']]

    wv = Word2Vec(sentences=data, min_count=1)
    print(wv.most_similar('makan', topn=5))
