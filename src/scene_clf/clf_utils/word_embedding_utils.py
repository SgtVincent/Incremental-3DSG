import gensim
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
import pickle


keyword_mapping={
    "coatrack":"coat rack",
    "hatrack":"hat rack",
    "bedframe":"bed frame",
    "stepstool":"step stool"
}

def create_embedding(words, save_path, model='glove-twitter-25'):
    # refer to https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models
    # for more pretrained models
    mapping = {}

    # TODO: improve the word embedding methods
    glove_vectors = gensim.downloader.load(model)
    for word in words:
        # naive method for multi-gram word: weighted sum
        if word in keyword_mapping:
            mapped_word = keyword_mapping[word]
        else:
            mapped_word = word

        weight = 0
        vec = np.zeros(glove_vectors.vector_size)
        word_l = mapped_word.split()
        for simple_word in word_l:

            if simple_word in glove_vectors:
                weight = weight + 1.0
                vec = vec + glove_vectors[simple_word]
            else: # find the closest synonym
                result = glove_vectors.most_similar(simple_word)
                key, sim = result[0]
                weight = weight + sim
                vec = vec + glove_vectors[key]
        vec = vec / weight
        mapping[word] = vec
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping


# def get_embedding(word, glove_dict):
#     if word in keyword_mapping:
#         word = keyword_mapping[word]
#     return glove_dict[word]
