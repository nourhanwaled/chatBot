from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response based on the given prompt.

        :param prompt: The input string to generate a response for.
        :return: A generated response as a string.
        """
        pass
