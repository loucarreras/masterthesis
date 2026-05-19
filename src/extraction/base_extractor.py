from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, text: str):
        pass