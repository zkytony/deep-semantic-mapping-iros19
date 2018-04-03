# author: Kaiyu Zheng

from abc import ABC, abstractmethod

class Experiment(ABC):

    def __init__(self, root_dir="~", name="Unnamed"):
        self._name = name

    @abstractmethod
    def load_training_data(self, *args, **kwargs):
        pass


    @abstractmethod
    def load_testing_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_stats(self, *args, **kwargs):
        pass


class TestCase(ABC):

    def __init__(self, experiment):
        self._experiment = experiment

    @abstractmethod
    def run(self, sess, *args, **kwargs):
        pass

    @abstractmethod
    def _report(self):
        """
        Reports some custom results right after the test case finishes.
        """
        pass

    @abstractmethod
    def save_results(self, **kwargs):
        """
        Saves the reported results as well as all necessary data during
        the testing.

        Returns a report (dict).

        **kwargs

           path (str): path to saved results (Required).
        """
        pass

    @classmethod
    def name(cls):
        if cls.__name__.startswith("TestCase_"):
            return cls.__name__.split("_")[1]
        else:
            return cls.__name__
