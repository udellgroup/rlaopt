"""Utilities for counting the number of ids.

Taken from https://github.com/cvxpy/cvxpy/blob/master/cvxpy/lin_ops/lin_utils.py.
"""


class Counter:
    """A counter for ids.

    Attributes:
        count (int): The current count, starting from 1.
    """

    def __init__(self) -> None:
        """Initializes the counter with a count of 1."""
        self.count = 1


ID_COUNTER = Counter()


def get_id() -> int:
    """Returns a new id and increments the id counter.

    Returns:
        int: A new id.
    """
    new_id = ID_COUNTER.count
    ID_COUNTER.count += 1
    return new_id
