# abstract class that all the balance measures are defined by

from abc import ABC, abstractmethod


class BalanceMeasure(ABC):
    @abstractmethod
    def measures(self):
        pass
