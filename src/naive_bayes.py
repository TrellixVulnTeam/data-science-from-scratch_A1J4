from collections import defaultdict, Counter
import glob
import math
import os
import random
import re
from typing import (
    Set, 
    NamedTuple,
    List, 
    Tuple, 
    Dict, 
    Iterable
)

from machine_learning_intro import split_data

# Creating spam or not spam filter using Naive Bayes


def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)                       # remove duplicates.


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0


    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1


    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham


    def predict(self, text: str) -> float:
        # taking into account that p_1 * ... * p_n = exp(log(p_1) + ... + log(p_n))
        # as floating points multiplication doesn't work well in case when
        # we have numbers close to 0
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it;
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


def download_data(output_dir: str):
    from io import BytesIO
    import requests
    import tarfile

    # So we can treat bytes as a file.
    # To download the files, which
    # are in .tar.bz format.
    BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
    FILES = ["20021010_easy_ham.tar.bz2",
             "20021010_hard_ham.tar.bz2",
             "20021010_spam.tar.bz2"]
    # This is where the data will end up,
    # in /spam, /easy_ham, and /hard_ham subdirectories.
    # Change this to where you want the data.
    for filename in FILES:
        # Use requests to get the file contents at each URL.
        content = requests.get(f"{BASE_URL}/{filename}").content
        # Wrap the in-memory bytes so we can use them as a "file."
        fin = BytesIO(content)
        # And extract all the files to the specified output dir.
        with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
           tf.extractall(output_dir)


def preprocess_data(path: str):
    path_pattern = f"{path}/*/*"
    data: List[Message] = []

    # glob.glob returns every filename that matches the wildcarded path
    for filename in glob.glob(path_pattern):
        is_spam = "ham" not in filename
    
        # There are some garbage characters in the emails, the errors='ignore'
        # skips them instead of raising an exception.
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  # done with this file

    return data


def main():
    path = "data/spam_data"

    if not os.path.isdir(path):
        download_data(path)

    data = preprocess_data(path)
    
    random.seed(0)      # just so you get the same answers as me
    train_messages, test_messages = split_data(data, 0.75)
    
    model = NaiveBayesClassifier()
    model.train(train_messages)

    predictions = [(message, model.predict(message.text))
                   for message in test_messages]


    # Assume that spam_probability > 0.5 corresponds to spam prediction
    # and count the combinations of (actual is_spam, predicted is_spam)
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)

    # print(f"accuracy: {pct_correct}")
    print("confusion matrix: (actual, predicted)")
    for pair, count in confusion_matrix.items():
        print(f"{pair}: {count}")

    precision = confusion_matrix[(True, True)] /  (confusion_matrix[(True, True)] + confusion_matrix[(False, True)])
    recall = confusion_matrix[(True, True)] /  (confusion_matrix[(True, True)] + confusion_matrix[(True, False)])
    precision = round(precision, 2)
    recall = round(recall, 2)
    print(f"precision: {precision}")
    print(f"recall: {recall}")


if __name__ == "__main__":
   main()