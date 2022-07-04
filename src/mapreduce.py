import datetime
import calendar
from typing import (
    Any,
    List,
    Tuple, 
    Iterator, 
    Iterable,
    Callable,
    NamedTuple
)
from collections import defaultdict, Counter


# count words in documents using mapreduce
def tokenize(document: str) -> List[str]:
    """Just split on whitespace"""
    return document.split()


def wc_mapper(document: str) -> Iterator[Tuple[str, int]]:
    """For each word in the document, emit (word, 1)"""
    for word in tokenize(document):
        yield (word, 1)


def wc_reducer(word: str,
               counts: Iterable[int]) -> Iterator[Tuple[str, int]]:
    """Sum up the counts for a word"""
    yield (word, sum(counts))


def word_count(documents: List[str]) -> List[Tuple[str, int]]:
    """Count the words in the input documents using MapReduce"""

    collector = defaultdict(list)  # To store grouped values

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.items()
            for output in wc_reducer(word, counts)]


# making more general map_reduce function

# A key-value pair is just a 2-tuple
KV = Tuple[Any, Any]

# A Mapper is a function that returns an Iterable of key-value pairs
Mapper = Callable[..., Iterable[KV]]

# A Reducer is a function that takes a key and an iterable of values
# and returns a key-value pair
Reducer = Callable[[Any, Iterable], Iterable[KV]]


def map_reduce(inputs: Iterable,
               mapper: Mapper,
               reducer: Reducer) -> List[KV]:
    """Run MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.items()
            for output in reducer(key, values)]


def values_reducer(values_fn: Callable) -> Reducer:
    """Return a reducer that just applies values_fn to its values"""
    def reduce(key, values: Iterable) -> Iterable[KV]:
        yield (key, values_fn(values))

    return reduce


def data_science_day_mapper(status_update: dict) -> Iterable:
        """Yields (day_of_week, 1) if status_update contains "data science" """
        if "data science" in status_update["text"].lower():
            day_of_week = calendar.day_name[status_update["created_at"].weekday()]
            yield (day_of_week, 1)


def words_per_user_mapper(status_update: dict):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
        yield (user, (word, 1))


def most_popular_word_reducer(user: str,
                                words_and_counts: Iterable[KV]):
    """
    Given a sequence of (word, count) pairs,
    return the word with the highest total count
    """
    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count

    word, count = word_counts.most_common(1)[0]

    yield (user, (word, count))


def liker_mapper(status_update: dict):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)


# Matrix multiplication

# More compact representation for sparse matrices 
class Entry(NamedTuple):
    name: str
    i: int
    j: int
    value: float 


def matrix_multiply_mapper(num_rows_a: int, num_cols_b: int) -> Mapper:
    # C[x][y] = A[x][0] * B[0][y] + ... + A[x][m] * B[m][y]
    #
    # so an element A[i][j] goes into every C[i][y] with coef B[j][y]
    # and an element B[i][j] goes into every C[x][j] with coef A[x][i]
    def mapper(entry: Entry):
        if entry.name == "A":
            for y in range(num_cols_b):
                key = (entry.i, y)              # which element of C
                value = (entry.j, entry.value)  # which entry in the sum
                yield (key, value)
        else:
            for x in range(num_rows_a):
                key = (x, entry.j)              # which element of C
                value = (entry.i, entry.value)  # which entry in the sum
                yield (key, value)

    return mapper


def matrix_multiply_reducer(key: Tuple[int, int],
                            indexed_values: Iterable[Tuple[int, int]]):
    results_by_index = defaultdict(list)

    for index, value in indexed_values:
        results_by_index[index].append(value)

    # Multiply the values for positions with two values
    # (one from A, and one from B) and sum them up.
    sumproduct = sum(values[0] * values[1]
                     for values in results_by_index.values()
                     if len(values) == 2)

    if sumproduct != 0.0:
        yield (key, sumproduct)


def main():
    documents = ["data science", "big data", "science fiction"]

    print("Count words in documents using mapreduce:")
    word_counts = word_count(documents)
    print(word_counts)
    print()

    print("Same as above except using general map_reduce function:")
    word_counts = map_reduce(documents, wc_mapper, wc_reducer)
    print(word_counts)
    print()

    sum_reducer = values_reducer(sum)
    max_reducer = values_reducer(max)
    min_reducer = values_reducer(min)
    count_distinct_reducer = values_reducer(lambda values: len(set(values)))

    print("Some reducers using general values_reducer function:")
    print(f"sum_reducer('key', [1, 2, 3, 3]) = {next(iter(sum_reducer('key', [1, 2, 3, 3])))}")
    print(f"min_reducer('key', [1, 2, 3, 3]) = {next(iter(min_reducer('key', [1, 2, 3, 3])))}")
    print(f"max_reducer('key', [1, 2, 3, 3]) = {next(iter(max_reducer('key', [1, 2, 3, 3])))}")
    print(f"count_distinct_reducer('key', [1, 2, 3, 3]) = {next(iter(count_distinct_reducer('key', [1, 2, 3, 3])))}")
    print()

    print("Analyzing status updates")
    status_updates = [
        {"id": 2,
         "username" : "joelgrus",
         "text" : "Should I write a second edition of my data science book?",
         "created_at" : datetime.datetime(2018, 2, 21, 11, 47, 0),
         "liked_by" : ["data_guy", "data_gal", "mike"] },
         # ...
    ]

    print(f"status_updates = {status_updates}")
    print()

    print("Count how many data science updates there are on each day of the week")
    data_science_days = map_reduce(status_updates,
                                   data_science_day_mapper,
                                   sum_reducer)
    print(data_science_days)
    print()

    print("Find out for each user the most common word that she puts in her status updates")
    user_words = map_reduce(status_updates,
                            words_per_user_mapper,
                            most_popular_word_reducer)
    print(user_words)
    print()

    print("Find out the number of distinct status-likers for each user")
    distinct_likers_per_user = map_reduce(status_updates,
                                          liker_mapper,
                                          count_distinct_reducer)
    print(distinct_likers_per_user)
    print()

    print("Sparce matrix multiplication")

    A = [[3, 2, 0],
        [0, 0, 0]]

    B = [[4, -1, 0],
        [10, 0, 0],
        [0, 0, 0]]

    # Sparce representation of A and B
    entries = [Entry("A", 0, 0, 3), Entry("A", 0, 1,  2), Entry("B", 0, 0, 4),
               Entry("B", 0, 1, -1), Entry("B", 1, 0, 10)]
    
    mapper = matrix_multiply_mapper(num_rows_a=2, num_cols_b=3)
    reducer = matrix_multiply_reducer
    # A x B = C
    C = map_reduce(entries, mapper, reducer) 

    print(f"A = {[((entry.i, entry.j), entry.value) for entry in entries if entry.name == 'A']}")
    print(f"B = {[((entry.i, entry.j), entry.value) for entry in entries if entry.name == 'B']}")
    print(f"A x B = {C}")


if __name__ == "__main__":
    main()