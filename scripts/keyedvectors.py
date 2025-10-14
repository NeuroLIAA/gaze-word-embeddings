import numpy as np
from typing import List, Union


class KeyedVectors:
    """
    A lightweight replacement for gensim.models.KeyedVectors that efficiently
    loads Word2Vec format embeddings from plain text files.
    """
    def __init__(self):
        self.vectors = None
        self.vocab = {}  # word -> index mapping
        self.index_to_word = []
        self.vector_size = 0
        
    @classmethod
    def load_word2vec_format(cls, filepath: str, binary: bool = False,
                             limit: int = None, encoding: str = 'utf-8'):
        """
        Load word embeddings from a Word2Vec format file.
        
        Args:
            filepath: Path to the embeddings file
            binary: If True, load binary format (not implemented)
            limit: Maximum number of word vectors to load
            encoding: File encoding (default: utf-8)
            
        Returns:
            KeyedVectors instance with loaded embeddings
        """
        if binary:
            raise NotImplementedError("Binary format not supported")
            
        kv = cls()
        
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            # Read header
            first_line = f.readline().strip().split()
            vocab_size = int(first_line[0])
            vector_size = int(first_line[1])
            
            if limit:
                vocab_size = min(vocab_size, limit)
            
            kv.vector_size = vector_size
            kv.vectors = np.zeros((vocab_size, vector_size), dtype=np.float32)
            
            # Read vectors
            for i in range(vocab_size):
                line = f.readline()
                if not line:
                    break
                    
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                kv.vocab[word] = i
                kv.index_to_word.append(word)
                kv.vectors[i] = vector
        
        return kv
    
    def __getitem__(self, words: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for one or more words.
        
        Args:
            words: A single word (str) or list of words
            
        Returns:
            numpy array of shape (embedding_dim,) for single word
            or (num_words, embedding_dim) for multiple words
        """
        if isinstance(words, str):
            if words not in self.vocab:
                raise KeyError(f"Word '{words}' not in vocabulary")
            return self.vectors[self.vocab[words]]
        
        # Handle list of words
        indices = []
        for word in words:
            if word not in self.vocab:
                raise KeyError(f"Word '{word}' not in vocabulary")
            indices.append(self.vocab[word])
        
        return self.vectors[indices]
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        if word1 not in self.vocab:
            raise KeyError(f"Word '{word1}' not in vocabulary")
        if word2 not in self.vocab:
            raise KeyError(f"Word '{word2}' not in vocabulary")
        
        vec1 = self.vectors[self.vocab[word1]]
        vec2 = self.vectors[self.vocab[word2]]
        
        # Compute cosine similarity: (vec1 · vec2) / (||vec1|| * ||vec2||)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word in self.vocab
    
    def __len__(self) -> int:
        """Return the number of words in the vocabulary."""
        return len(self.vocab)
    
    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """
        Find the most similar words to a given word.
        
        Args:
            word: Query word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        word_vec = self.vectors[self.vocab[word]]
        
        # Compute cosine similarities with all words
        # cosine_sim = (word_vec · all_vecs) / (||word_vec|| * ||all_vecs||)
        dot_products = np.dot(self.vectors, word_vec)
        word_norm = np.linalg.norm(word_vec)
        all_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        denominators = word_norm * all_norms
        denominators[denominators == 0] = 1
        similarities = dot_products / denominators
        # Get top-n indices (excluding the query word itself)
        word_idx = self.vocab[word]
        similarities[word_idx] = -np.inf  # Exclude the word itself
        
        top_indices = np.argsort(similarities)[::-1][:topn]
        return [(self.index_to_word[idx], float(similarities[idx])) 
                for idx in top_indices]
