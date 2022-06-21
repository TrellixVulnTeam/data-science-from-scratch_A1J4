import re
import random
import requests
from collections import defaultdict
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup


def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")


def generate_using_bigrams(transitions: Dict[str, List[str]]) -> str:
    current = "."   # this means the next word will start a sentence
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigrams (current, _)
        current = random.choice(next_word_candidates)  # choose one at random
        result.append(current)                         # append it to results
        if current == ".": return " ".join(result)     # if "." we're done


def generate_using_trigrams(transitions: Dict[Tuple[str, str], List[str]], starts: List[str]) -> str:
    current = random.choice(starts)   # choose a random starting word
    prev = "."                        # and precede it with a '.'
    result = [current]
    while True:
        next_word_candidates = transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result)


def main():
    url = "https://www.oreilly.com/ideas/what-is-data-science"
    html = requests.get(url)
    html.encoding = "utf-8" # changing encoding as wrong one has been guessed
    soup = BeautifulSoup(html.text, 'html5lib')

    content = soup.find("div", "main-post-radar-content")   # find main-post-radar-content div
    regex = r"[\w']+|[\.]"                                  # matches a word or a period

    document = []

    if not content:
        raise ValueError("content is None")

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    # bigram transitions
    bigram_transitions = defaultdict(list)
    for prev, current in zip(document, document[1:]):
        bigram_transitions[prev].append(current)
    
    print("-- Generating text using bigram model: ")
    print(generate_using_bigrams(bigram_transitions))
    print()
    
    # trigram transitions
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in zip(document, document[1:], document[2:]):

        if prev == ".":              # if the previous "word" was a period
            starts.append(current)   # then this is a start word

        trigram_transitions[(prev, current)].append(next)

    print("-- Generating text using trigram model: ")
    print(generate_using_trigrams(trigram_transitions, starts))


if __name__ == "__main__":
    main()