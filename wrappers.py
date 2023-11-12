from abc import abstractmethod


class ModelWrapper:
    def __init__(self):
        pass

    @staticmethod
    def load(model_dir: str) -> 'ModelWrapper':
        pass

    @abstractmethod
    def predict(self, text_input: str, settings: dict) -> str:
        pass


class LlamaModel:
    _CURRENT_MODEL = None

    @staticmethod
    def load_model(model_class: type[ModelWrapper], model_dir) -> None:
        LlamaModel._CURRENT_MODEL = model_class.load(model_dir)

    @staticmethod
    def predict(text_input: str, settings: dict) -> str:
        return LlamaModel._CURRENT_MODEL.predict(text_input, settings).replace(text_input, ' ')
