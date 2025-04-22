from abc import ABC, abstractmethod

__all__ = ["Solver"]


class Solver(ABC):
    def __init__(self, *args: list, **kwargs: dict):
        pass

    @abstractmethod
    def _get_precond(self, *args: list, **kwargs: dict):
        pass

    @abstractmethod
    def _step(self, *args: list, **kwargs: dict):
        pass
