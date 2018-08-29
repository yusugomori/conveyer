from ..Model import Model


class ClassificationModel(Model):
    def __init__(self):
        super().__init__()
        self.metrics = ['accuracy', 'f1']
