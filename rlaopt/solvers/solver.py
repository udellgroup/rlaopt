from abc import ABC


class Solver(ABC):
    def __init__(self, *args: list, **kwargs: dict):
        pass

    def _get_precond(self):
        pass

    def _step(self, *args: list, **kwargs: dict):
        pass
