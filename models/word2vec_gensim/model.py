from gensim.models import Word2Vec

class Word2vecGensim:
    def __init__(self, corpora, vector_size, window_size, min_count, negative_samples, epochs, cbow, model_name, save_path):
        self.corpora = corpora
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.epochs = epochs
        self.cbow = cbow
        self.model_name = model_name
        self.save_path = save_path

    def train(self):
        model = Word2Vec(sentences=self.corpora, sg=not self.cbow, vector_size=self.vector_size, window=self.window_size, min_count=self.min_count,
                         negative=self.negative_samples, epochs=self.epochs, workers=12)
        self.save_path.mkdir(exist_ok=True, parents=True)
        model.wv.save_word2vec_format(str(self.save_path / f'{self.model_name}.vec'))