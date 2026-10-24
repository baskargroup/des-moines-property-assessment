from abc import ABC, abstractmethod

class BaseVLM(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def analyze(self, image_path, prompt):
        """
        Sends an image and prompt to the VLM.
        
        Args:
            image_path (str): Path to the image file.
            prompt (str): The text prompt.
            
        Returns:
            str: The raw text response from the model.
        """
        pass

