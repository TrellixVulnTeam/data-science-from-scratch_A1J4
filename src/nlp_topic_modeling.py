import random
from typing import List
from collections import Counter

import tqdm


documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4   # trying to find 4 topics

# a list of Counters, one for each document
document_topic_counts = [Counter() for _ in documents]

# a list of Counters, one for each topic
topic_word_counts = [Counter() for _ in range(K)]

# a list of numbers, one for each topic
topic_counts = [0 for _ in range(K)]

# a list of numbers, one for each document
document_lengths = [len(document) for document in documents]

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

D = len(documents)


def sample_from(weights: List[float]) -> int:
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()      # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w                       # return the smallest i such that
        if rnd <= 0: 
            return i          # weights[0] + ... + weights[i] >= rnd

    raise ValueError("Something went wrong, this should never happen")


def p_topic_given_document(topic: int, d: int, alpha: float = 0.1) -> float:
    """
    The fraction of words in document _d_
    that are assigned to _topic_ (plus some smoothing)
    """
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))


def p_word_given_topic(word: str, topic: int, beta: float = 0.1) -> float:
    """
    The fraction of words assigned to _topic_
    that equal _word_ (plus some smoothing)
    """
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))


def topic_weight(d: int, word: str, k: int) -> float:
    """
    Given a document and a word in that document,
    return the weight for the kth topic
    """
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)


def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])


def main():
    random.seed(0)

    document_topics = [[random.randrange(K) for word in document]
                        for document in documents]

    for d in range(D):
        for word, topic in zip(documents[d], document_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1        

    for iter in tqdm.trange(10000):
        for d in range(D):
            for i, (word, topic) in enumerate(zip(documents[d],
                                                document_topics[d])):

                # remove this word / topic from the counts
                # so that it doesn't influence the weights
                document_topic_counts[d][topic] -= 1
                topic_word_counts[topic][word] -= 1
                topic_counts[topic] -= 1
                document_lengths[d] -= 1

                # choose a new topic based on the weights
                new_topic = choose_new_topic(d, word)
                document_topics[d][i] = new_topic

                # and now add it back to the counts
                document_topic_counts[d][new_topic] += 1
                topic_word_counts[new_topic][word] += 1
                topic_counts[new_topic] += 1
                document_lengths[d] += 1

    # for k, word_counts in enumerate(topic_word_counts):
    #     for word, count in word_counts.most_common():
    #         if count > 0:
    #             print(k, word, count)

    topic_names = ["Big Data and programming languages",
                   "Python and statistics",
                   "databases",
                   "machine learning"]

    for document, topic_count in zip(documents, document_topic_counts):
        print(document)
        for topic, count in topic_count.most_common():
            if count > 0:
                print(topic_names[topic], count)
        print()


if __name__ == "__main__":
    main()