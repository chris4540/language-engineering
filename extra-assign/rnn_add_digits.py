import numpy as np


class RecurrentUnit:
    """
    Simple RNN unit for summing up digits
    """

    def __init__(self):
        self.w_hh = np.ones(shape=(1))  # map the h_{t-1} to h_{t}
        self.w_xh = np.ones(shape=(1))  # map input to h_{t}
        self.w_ho = np.ones(shape=(1))  # map h_{t} to output

    def forward(self, input_: int, hidden_state: int = 0):
        # use linear activation
        h = self.w_xh.dot(input_) + self.w_hh.dot(hidden_state)

        output = self.w_ho.dot(h)
        return h, output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def sum_with_rnn_unit(number: int):
    unit = RecurrentUnit()
    h = 0  # init the hidden state to be zero
    for d in str(number):
        h, output = unit(int(d), h)
    return output


if __name__ == "__main__":
    for n in range(1000, 10000):
        expect = sum([int(d) for d in str(n)])
        outcome = sum_with_rnn_unit(n)
        assert expect == outcome, f"{expect} != {outcome}"

    print("Test pass!")
