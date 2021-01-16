from pymoo.model.callback import Callback
from Pymoo.algorithm import algorithm


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.gen = 0
        self.data["best_obj1"] = []
        self.data['best_obj2'] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        self.data["best_obj1"].append(algorithm.pop.get("F")[:, 0].min())
        self.data['best_obj2'].append(algorithm.pop.get('F')[:, 1].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        self.gen += 1


callback = MyCallback()
