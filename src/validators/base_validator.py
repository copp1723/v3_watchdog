# src/validators/base_validator.py

class BaseValidator:
    def __init__(self, data):
        self.data = data

    def validate(self):
        raise NotImplementedError("Subclasses must implement validate().")