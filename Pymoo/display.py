from pymoo.util.display import Display


class MyDisplay(Display):
    bestObj = []

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        # self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
        # self.output.append("metric_b", np.mean(algorithm.pop.get("F")))
        self.output.append('Best Lift [N]', np.mean(algorithm.pop.get("F")[:, 0].min()))
        self.output.append('Best Drag [N]', np.mean(algorithm.pop.get("F")[:, 1].min()))

        # if


display = MyDisplay()
