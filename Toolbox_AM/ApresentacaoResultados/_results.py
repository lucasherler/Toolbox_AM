#import pandas
import abc

class Result(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def generate(dataframe):
        pass

    @staticmethod
    @abc.abstractmethod
    def getName():
        pass
