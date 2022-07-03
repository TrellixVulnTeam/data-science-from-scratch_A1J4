import re
import csv
import random
from typing import (
    Dict, 
    List, 
    Optional,
    NamedTuple 
)
from collections import defaultdict
import tqdm

from basics.linear_algebra import dot
from basics.tensor import Tensor, random_tensor
from working_with_data_pca import pca, transform

# Path to the data
MOVIES = "data/movie_ratings/u.item"   # pipe-delimited: movie_id|title|...
RATINGS = "data/movie_ratings/u.data"  # tab-delimited: user_id, movie_id, rating, timestamp


class Rating(NamedTuple):
    user_id: str
    movie_id: str
    rating: float


def print_avg_ratings(title_pattern: str,
                      movies: Dict[str, str],
                      ratings: List[Rating]):
    # Data structure for accumulating ratings by movie_id
    star_wars_ratings = {movie_id: []
                         for movie_id, title in movies.items()
                         if re.search(title_pattern, title)}
    
    # Iterate over ratings, accumulating the Star Wars ones
    for rating in ratings:
        if rating.movie_id in star_wars_ratings:
            star_wars_ratings[rating.movie_id].append(rating.rating)
    
    # Compute the average rating for each movie
    avg_ratings = [(sum(title_ratings) / len(title_ratings), movie_id)
                   for movie_id, title_ratings in star_wars_ratings.items()]
    
    # And then print them in order
    for avg_rating, movie_id in sorted(avg_ratings, reverse=True):
        print(f"{avg_rating:.2f} {movies[movie_id]}")
    

def loop(dataset: List[Rating],
         user_vectors: Dict[str, Tensor],
         movie_vectors: Dict[str, Tensor],
         embedding_dim: int,
         learning_rate: Optional[float] = None) -> None:
    
    stage = "train" if learning_rate is not None else "eval"

    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i, rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = dot(user_vector, movie_vector)
            error = predicted - rating.rating
            loss += error ** 2

            if stage == "train":
                #     predicted = m_0 * u_0 + ... + m_k * u_k
                # So each u_j enters output with coefficent m_j
                # and each m_j enters output with coefficient u_j
                user_gradient = [error * m_j for m_j in movie_vector]
                movie_gradient = [error * u_j for u_j in user_vector]

                # Take gradient steps
                for j in range(embedding_dim):
                    user_vector[j] -= learning_rate * user_gradient[j]
                    movie_vector[j] -= learning_rate * movie_gradient[j]

            t.set_description(f"({stage}) avg loss: {loss / (i + 1)}")


def main():
    
    random.seed(0)

    # We specify this encoding to avoid a UnicodeDecodeError.
    # see: https://stackoverflow.com/a/53136168/1076346
    with open(MOVIES, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="|")
        movies = {movie_id: title for movie_id, title, *_ in reader}
    

    # Create a list of [Rating]
    with open(RATINGS, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="\t")
        ratings = [Rating(user_id, movie_id, float(rating))
                   for user_id, movie_id, rating, _ in reader]

    print("Average ratings for some movies")
    print_avg_ratings("Star Wars|Empire Strikes|Jedi", movies, ratings)
    print()

    random.shuffle(ratings)
    
    split1 = int(len(ratings) * 0.7)
    split2 = int(len(ratings) * 0.85)
    
    train = ratings[:split1]              # 70% of the data
    validation = ratings[split1:split2]   # 15% of the data
    test = ratings[split2:]               # 15% of the data

    # we will use mse as a metric 
    # baseline error
    avg_rating = sum(rating.rating for rating in train) / len(train)
    baseline_error = sum((rating.rating - avg_rating) ** 2
                         for rating in test) / len(test)
    
    print("Baseline model that predicts the average rating")
    print(f"Baseline error: {baseline_error}")
    print()

    # Embedding vectors for matrix factorization model
    EMBEDDING_DIM = 2
    
    # Find unique ids
    user_ids = {rating.user_id for rating in ratings}
    movie_ids = {rating.movie_id for rating in ratings}
    
    # Then create a random vector per id
    user_vectors = {user_id: random_tensor(EMBEDDING_DIM)
                    for user_id in user_ids}
    movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM)
                     for movie_id in movie_ids}
    
    learning_rate = 0.05
    for epoch in range(6):
        learning_rate *= 0.9
        print(f"epoch: {epoch}, learning rate: {learning_rate}")
        loop(train, user_vectors, movie_vectors, EMBEDDING_DIM, learning_rate=learning_rate)
        loop(validation, user_vectors, movie_vectors, EMBEDDING_DIM)
    loop(test, user_vectors, movie_vectors, EMBEDDING_DIM)
    print()

    print("Using pca to explore components")
    component_number = 2 
    original_vectors = [vector for vector in movie_vectors.values()]
    components = pca(original_vectors, component_number)

    ratings_by_movie = defaultdict(list)
    for rating in ratings:
        ratings_by_movie[rating.movie_id].append(rating.rating)
    
    vectors = [
        (movie_id,
         sum(ratings_by_movie[movie_id]) / len(ratings_by_movie[movie_id]),
         movies[movie_id],
         vector)
        for movie_id, vector in zip(movie_vectors.keys(),
                                    transform(original_vectors, components))
    ]
    
    print()

    # Print top 25 and bottom 25 by first principal component
    top_25 = sorted(vectors, key=lambda v: v[-1][0])[:25]
    bottom_25 = sorted(vectors, key=lambda v: v[-1][0])[-25:]

    print(f"Top 25 movies based on {component_number} principal components")
    print()
    for movie in top_25:
        print(movie)
    print()

    print(f"Bottom 25 movies based on {component_number} principal components")
    print()
    for movie in bottom_25:
        print(movie)
    print()


if __name__ == "__main__":
    main()