import random
import requests
from typing import Iterable

import tqdm
from bs4 import BeautifulSoup

from deep_learning import (
    tanh, 
    Layer,
    Linear,
    softmax,
    Momentum,
    Sequential,
    SoftmaxCrossEntropy
)
from basics.linear_algebra import dot
from basics.tensor import (
    Tensor, 
    tensor_apply, 
    random_tensor
)
from nlp_deep_learning import Vocabulary
from nlp_topic_modeling import sample_from


class SimpleRnn(Layer):
    """Just about the simplest possible recurrent layer."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w = random_tensor(hidden_dim, input_dim, init='xavier')
        self.u = random_tensor(hidden_dim, hidden_dim, init='xavier')
        self.b = random_tensor(hidden_dim)

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]

    def forward(self, input: Tensor) -> Tensor:
        self.input = input              # Save both input and previous
        self.prev_hidden = self.hidden  # hidden state to use in backprop.

        a = [(dot(self.w[h], input) +           # weights @ input
              dot(self.u[h], self.hidden) +     # weights @ hidden
              self.b[h])                        # bias
             for h in range(self.hidden_dim)]

        self.hidden = tensor_apply(tanh, a)  # Apply tanh activation
        return self.hidden                   # and return the result.

    def backward(self, gradient: Tensor):
        # Backpropagate through the tanh
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]

        # b has the same gradient as a
        self.b_grad = a_grad

        # Each w[h][i] is multiplied by input[i] and added to a[h],
        # so each w_grad[h][i] = a_grad[h] * input[i]
        self.w_grad = [[a_grad[h] * self.input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]

        # Each u[h][h2] is multiplied by hidden[h2] and added to a[h],
        # so each u_grad[h][h2] = a_grad[h] * prev_hidden[h2]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]

        # Each input[i] is multiplied by every w[h][i] and added to a[h],
        # so each input_grad[i] = sum(a_grad[h] * w[h][i] for h in ...)
        return [sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]


def main():
    url = "https://www.ycombinator.com/topcompanies/"
    soup = BeautifulSoup(requests.get(url).text, 'html5lib')
    
    # We get the companies twice, so use a set comprehension to deduplicate.
    companies = list({a.text
                      for a in soup("a")
                      if "company-name" in a.get("class", ())})

    vocab = Vocabulary([c for company in companies for c in company])

    START = "^"
    STOP = "$"

    # We need to add them to the vocabulary too.
    vocab.add(START)
    vocab.add(STOP)

    HIDDEN_DIM = 32  # You should experiment with different sizes!
    
    rnn1 =  SimpleRnn(input_dim=vocab.size, hidden_dim=HIDDEN_DIM)
    rnn2 =  SimpleRnn(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
    linear = Linear(input_dim=HIDDEN_DIM, output_dim=vocab.size)
    
    model = Sequential([
        rnn1,
        rnn2,
        linear
    ])

    def generate(seed: str = START, max_len: int = 50) -> str:
        rnn1.reset_hidden_state()  # Reset both hidden states.
        rnn2.reset_hidden_state()
        output = [seed]            # Start the output with the specified seed.
    
        # Keep going until we produce the STOP character or reach the max length
        while output[-1] != STOP and len(output) < max_len:
            # Use the last character as the input
            input = vocab.one_hot_encode(output[-1])
    
            # Generate scores using the model
            predicted = model.forward(input)
    
            # Convert them to probabilities and draw a random char_id
            probabilities = softmax(predicted)
            next_char_id = sample_from(probabilities)
    
            # Add the corresponding char to our output
            output.append(vocab.get_word(next_char_id))
    
        # Get rid of START and END characters and return the word.
        return ''.join(output[1:-1])


    loss = SoftmaxCrossEntropy()
    optimizer = Momentum(learning_rate=0.01, momentum=0.9)
    
    for epoch in range(300):
        random.shuffle(companies)  # Train in a different order each epoch.
        epoch_loss = 0             # Track the loss.
        for company in tqdm.tqdm(companies):
            rnn1.reset_hidden_state()  # Reset both hidden states.
            rnn2.reset_hidden_state()
            company = START + company + STOP   # Add START and STOP characters.
    
            # The rest is just our usual training loop, except that the inputs
            # and target are the one-hot-encoded previous and next characters.
            for prev, next in zip(company, company[1:]):
                input = vocab.one_hot_encode(prev)
                target = vocab.one_hot_encode(next)
                predicted = model.forward(input)
                epoch_loss += loss.loss(predicted, target)
                gradient = loss.gradient(predicted, target)
                model.backward(gradient)
                optimizer.step(model)
    
        # Each epoch print the loss and also generate a name
        print(epoch, epoch_loss, generate())
    
        # Turn down the learning rate for the last 100 epochs.
        # There's no principled reason for this, but it seems to work.
        if epoch == 200:
            optimizer.lr *= 0.1

    # generating some names
    for _ in range(5):
        print(generate())


if __name__ == "__main__":
    main()