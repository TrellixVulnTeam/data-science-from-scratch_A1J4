import random
from typing import Dict, List


Grammar = Dict[str, List[str]]


def is_terminal(token: str) -> bool:
    return token[0] != "_"


def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        # If this is a terminal token, skip it.
        if is_terminal(token): continue

        # Otherwise, it's a non-terminal token,
        # so we need to choose a replacement at random.
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # Replacement could be e.g. "_NP _VP", so we need to
            # split it on spaces and splice it in.
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        # Now call expand on the new list of tokens.
        return expand(grammar, tokens)

    # If we get here we had all terminals and are done
    return tokens


def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ["_S"])


def main():
    grammar: Grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                "_A _NP _P _A _N"],
        "_VP" : ["_V",
                "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
    }

    print(" ".join(generate_sentence(grammar)))

if __name__ == "__main__":
    main()