from abc import ABC, abstractmethod
from typing import List

__all__ = ["Solver"]


class Solver(ABC):
    def __init__(self, *args: List, **kwargs: dict):
        pass

    @abstractmethod
    def _get_precond(self, *args: List, **kwargs: dict):
        pass

    @abstractmethod
    def _step(self, *args: List, **kwargs: dict):
        pass
