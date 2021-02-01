import os
import numpy as np


checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
algorithm = checkpoint

try:
    os.mkdir('./plots')
except OSError:
    print(OSError)

from pymoo.visualization.scatter import Scatter

# Design space
ps = algorithm.problem.pareto_set()  # use_cache=False, flatten=False)
plot = Scatter(title='Design Space')
plot.add(res.X)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.save('plots/design_space.png')

# Objective Space
pf = algorithm.problem.pareto_front()
plot = Scatter(title="Objective Space")
plot.add(res.F)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save('plots/obj_space.png')

# All design points
plot = Scatter(title='Entire Design Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['var'][i], label='GEN %i' % i)
plot.save('plots/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][i], label='GEN %i' % i)
plot.save('plots/entire_obj_space.png')

# import matplotlib.pyplot as plt
# plt.clf()
# plt.title('Entire Design Space')
# for i in range(len(algorithm.callback.data['var'])):
#     plt.scatter(algorithm.callback.data['var'][i][0], algorithm.callback.data['var'][i][1], label='GEN %i' % i)#, marker='o')
# plt.savefig('plots/matplot_design_space.png')
# plt.close()