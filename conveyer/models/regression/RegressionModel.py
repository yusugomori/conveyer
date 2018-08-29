from ..Model import Model


class RegressionModel(Model):
    def __init__(self):
        super().__init__()
        self.metrics = ['r2']
