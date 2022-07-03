from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from basics.linear_algebra import Vector, Matrix
from nlp_deep_learning import cosine_similarity


users_interests = [
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

unique_interests = sorted({interest
                        for user_interests in users_interests
                        for interest in user_interests})


# Just recommend the popular once that is not
# yet an in an interests of that user
def most_popular_new_interests(
        user_interests: List[str],
        max_results: int = 5) -> List[Tuple[str, int]]:

    popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

    suggestions = [(interest, frequency)
                   for interest, frequency in popular_interests.most_common()
                   if interest not in user_interests]
    return suggestions[:max_results]


def make_user_interest_vector(user_interests: List[str]) -> Vector:
        """
        Given a list of interests, produce a vector whose ith element is 1
        if unique_interests[i] is in the list, 0 otherwise
        """
        return [1 if interest in user_interests else 0
                for interest in unique_interests]


def make_user_interest_matrix() -> Matrix:

    user_interest_matrix = [make_user_interest_vector(user_interests)
                             for user_interests in users_interests]
    
    return user_interest_matrix


def most_similar_users_to(user_id: int, user_interest_matrix: Matrix) -> List[Tuple[int, float]]:

    # Compute all the similarities
    user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                          for interest_vector_j in user_interest_matrix]
                         for interest_vector_i in user_interest_matrix]

    pairs = [(other_user_id, similarity)                      # Find other
             for other_user_id, similarity in                 # users with
                enumerate(user_similarities[user_id])         # nonzero
             if user_id != other_user_id and similarity > 0]  # similarity.

    return sorted(pairs,                                      # Sort them
                  key=lambda pair: pair[-1],                  # most similar
                  reverse=True)                               # first.


# User-based collaborative filtering
def user_based_suggestions(user_id: int,
                           include_current_interests: bool = False):

    # Create user interest matrix
    user_interest_matrix = make_user_interest_matrix()

    # Sum up the similarities.
    suggestions: Dict[str, float] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id, user_interest_matrix):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # Convert them to a sorted list.
    suggestions_sorted = sorted(suggestions.items(),
                                key=lambda pair: pair[-1],  # weight
                                reverse=True)

    # And (maybe) exclude already-interests
    if include_current_interests:
        return suggestions_sorted
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions_sorted
                if suggestion not in users_interests[user_id]]


def make_interest_user_matrix() -> Matrix:

    # Get user - interest matrix
    user_interest_matrix = make_user_interest_matrix()

    # transpose to get interest - user matrix
    interest_user_matrix = [[user_interest_vector[j]
                             for user_interest_vector in user_interest_matrix]
                             for j, _ in enumerate(unique_interests)]
    
    return interest_user_matrix


def most_similar_interests_to(interest_id: int, interest_user_matrix: Matrix):

    interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                              for user_vector_j in interest_user_matrix]
                             for user_vector_i in interest_user_matrix]

    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in 
                enumerate(interest_similarities[interest_id])
             if interest_id != other_interest_id and similarity > 0]
    
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)


def item_based_suggestions(user_id: int,
                           include_current_interests: bool = False):

    # Create user interest matrix
    user_interest_matrix = make_user_interest_matrix()
    
    # Create interest user matrix    
    interest_user_matrix = make_interest_user_matrix()
    
    # Add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id, interest_user_matrix)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # Sort them by weight
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


def main():
    
    user_id = 1
    print(f"User {user_id} current interests")
    print(users_interests[user_id])
    print()

    print(f"Most popular interests for user {user_id}")
    print(most_popular_new_interests(users_interests[user_id]))
    print()

    print(f"User-based collaborative filtering for user {user_id}")
    print(user_based_suggestions(user_id))
    print()

    print(f"Item-based collaborative filtering for user {user_id}")
    print(item_based_suggestions(user_id))
    print()


if __name__ == "__main__":
    main()