# the global factory method
from pymoo.factory import get_visualization
plot = get_visualization("scatter")

for gen in callback.data['obj']:
    #print(gen)
    plot.add(gen)

#plot.add(res.pop.get('F'), color="green", marker="x")
#print(res.pop.get('F'))
#plot.add(B, color="red", marker="*")
plot.show()

#########################################################################

from pymoo.visualization.scatter import Scatter

# get the pareto-set and pareto-front for plotting
ps = problem.pareto_set(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=False)

# Design Space
plot = Scatter(title = "Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
plot.apply(lambda ax: ax.set_ylim(-2, 2))
plot.show()

# Objective Space
plot = Scatter(title = "Objective Space")
plot.add(res.F)
plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.show()

#########################################################################

import matplotlib.pyplot as plt
matplotlib.use('GTK3Agg')

val = [e.pop.get("F").min() for e in res.history]
plt.plot(np.arange(len(val)), val)
plt.show()