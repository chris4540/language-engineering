import numpy as np
from typing import Iterable, List


class WordSample:
    def __init__(self, file, must_include_wordset: Iterable[str], n_samples=10000, random_state=40):
        with open(file, "r", encoding="utf-8") as f:
            self._words = f.read().splitlines()

        print(f"{file} has {len(self._words)} words.")
        avaliable = set(self._words).intersection(must_include_wordset)
        print("# of words intersect with `must_include_wordset`: ", len(avaliable))

        # select n_samples data
        rnd_state = np.random.RandomState(random_state)
        self._selected = rnd_state.choice(
            list(avaliable), size=n_samples, replace=False)
        assert len(set(self._selected)) == n_samples

    @property
    def words(self) -> List[str]:
        return self._selected
