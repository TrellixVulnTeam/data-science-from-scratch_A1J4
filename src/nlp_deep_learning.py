import re
import math
import json
import random
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Iterable
)

import matplotlib.pyplot as plt

from basics.tensor import (
    Tensor, 
    zeros_like,
    random_tensor 
)
from deep_learning import (
    Layer,
    Linear,
    Momentum,
    Sequential,
    GradientDescent,
    SoftmaxCrossEntropy
)
from working_with_data_pca import pca, transform
from basics.tensor import Tensor
from basics.linear_algebra import dot, Vector


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))


class Vocabulary:
    def __init__(self, words: Optional[List[str]] = None) -> None:
        self.w2i: Dict[str, int] = {}  # mapping word -> word_id
        self.i2w: Dict[int, str] = {}  # mapping word_id -> word

        for word in (words or []):     # If words were provided,
            self.add(word)             # add them.


    @property
    def size(self) -> int:
        """how many words are in the vocabulary"""
        return len(self.w2i)


    def add(self, word: str) -> None:
        if word not in self.w2i:        # If the word is new to us:
            word_id = len(self.w2i)     # Find the next id.
            self.w2i[word] = word_id    # Add to the word -> word_id map.
            self.i2w[word_id] = word    # Add to the word_id -> word map.


    def get_id(self, word: str) -> int:
        """return the id of the word (or None)"""
        id = self.w2i.get(word)
        if id is None:
            raise LookupError(f"Unknown word {word}")
        
        return id


    def get_word(self, word_id: int) -> str:
        """return the word with the given id (or None)"""
        word = self.i2w.get(word_id)
        if word is None:
            raise LookupError(f"Unknown word id {word_id}")

        return word


    def one_hot_encode(self, word: str) -> Tensor:
        word_id = self.get_id(word)

        return [1.0 if i == word_id else 0.0 for i in range(self.size)]


def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(vocab.w2i, f)       # Only need to save w2i


def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename) as f:
        # Load w2i and generate i2w from it.
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
    return vocab


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # One vector of size embedding_dim for each desired embedding
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)

        # Save last input id
        self.last_input_id = None


    def forward(self, input_id: int) -> Tensor:
        """Just select the embedding vector corresponding to the input id"""
        self.input_id = input_id    # remember for use in backpropagation

        return self.embeddings[input_id]


    def backward(self, gradient: Tensor) -> None:
        # Zero out the gradient corresponding to the last input.
        # This is way cheaper than creating a new all-zero tensor each time.
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row

        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient


    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]


    def grads(self) -> Iterable[Tensor]:
        return [self.grad]


class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        # Call the superclass constructor
        super().__init__(vocab.size, embedding_dim)

        # And hang onto the vocab
        self.vocab = vocab


    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        return self.embeddings[word_id]


    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        """Returns the n closest words based on cosine similarity"""
        vector = self[word]

        # Compute pairs (similarity, other_word), and sort most similar first
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)

        return scores[:n]


def main():
    random.seed(0)

    colors = ["red", "green", "blue", "yellow", "black", ""]
    nouns = ["bed", "car", "boat", "cat"]
    verbs = ["is", "was", "seems"]
    adverbs = ["very", "quite", "extremely", ""]
    adjectives = ["slow", "fast", "soft", "hard"]

    def make_sentence() -> str:
        return " ".join([
            "The",
            random.choice(colors),
            random.choice(nouns),
            random.choice(verbs),
            random.choice(adverbs),
            random.choice(adjectives),
            "."
        ])

    NUM_SENTENCES = 50
    
    sentences = [make_sentence() for _ in range(NUM_SENTENCES)]

    # This is not a great regex, but it works on our data.
    tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower())
                           for sentence in sentences]

    # Create a vocabulary (that is, a mapping word -> word_id) based on our text.
    vocab = Vocabulary([word
                        for sentence_words in tokenized_sentences
                        for word in sentence_words])
    
    inputs: List[int] = []
    targets: List[Tensor] = []
    
    for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):          # For each word
            for j in [i - 2, i - 1, i + 1, i + 2]:   # take the nearby locations
                if 0 <= j < len(sentence):           # that aren't out of bounds
                    nearby_word = sentence[j]        # and get those words.
    
                    # Add an input that's the original word_id
                    inputs.append(vocab.get_id(word))
    
                    # Add a target that's the one-hot-encoded nearby word
                    targets.append(vocab.one_hot_encode(nearby_word))


    # Model for learning word vectors
    EMBEDDING_DIM = 5  # seems like a good size
    
    # Define the embedding layer separately, so we can reference it.
    embedding = TextEmbedding(vocab=vocab, embedding_dim=EMBEDDING_DIM)
    
    model = Sequential([
        # Given a word (as a vector of word_ids), look up its embedding.
        embedding,
        # And use a linear layer to compute scores for "nearby words".
        Linear(input_dim=EMBEDDING_DIM, output_dim=vocab.size)
    ])

    loss = SoftmaxCrossEntropy()
    optimizer = GradientDescent(learning_rate=0.01)
    
    for epoch in range(100):
        epoch_loss = 0.0
        for input, target in zip(inputs, targets):
            predicted = model.forward(input)
            epoch_loss += loss.loss(predicted, target)
            gradient = loss.gradient(predicted, target)
            model.backward(gradient)
            optimizer.step(model)
        print(f"epoch: {epoch} loss: {epoch_loss}")             # Print the loss
        print(embedding.closest("black"))                       # and also a few nearest words
        print(embedding.closest("slow"))                        # so we can see what's being
        print(embedding.closest("car"))                         # learned.
    
    print()

    # Explore most similar words
    print("Top 5 most similar words")
    pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
             for w1 in vocab.w2i
             for w2 in vocab.w2i
             if w1 < w2]
    pairs.sort(reverse=True)
    print(pairs[:5])

    print()

    # Extract the first two principal components and transform the word vectors
    print("Extracting first 2 principal components using pca")
    components = pca(embedding.embeddings, 2)
    transformed = transform(embedding.embeddings, components)
    
    # Scatter the points (and make them white so they're "invisible")
    fig, ax = plt.subplots()
    ax.scatter(*zip(*transformed), marker='.', color='w')
    
    # Add annotations for each word at its transformed location
    for word, idx in vocab.w2i.items():
        ax.annotate(word, transformed[idx])
    
    # And hide the axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.show()


if __name__ == "__main__":
    main()