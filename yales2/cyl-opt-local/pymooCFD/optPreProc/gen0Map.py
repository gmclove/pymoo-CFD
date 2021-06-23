import os
import numpy as np
from pymooCFD.setupOpt import *

def gen0Map(heatmap=False):
    # checkpointDir = 'checkpoint' + '.npy'
    # checkpointFile = 'checkpoint.npy'
    plotDir = './plots'
    try:
        os.mkdir(plotDir)
    except OSError as err:
        print(err)
    ########################################################################################################################
    X, F = loadData()
    ########################################################################################################################
    print('VARS')
    print(X)
    print('OBJ')
    print(F)
    ########################################################################################################################
    ##### SCATTER PLOTS #######
    ###########################
    X = np.array(X)
    F = np.array(F)
    from pymoo.visualization.scatter import Scatter
    # https://pymoo.org/visualization/scatter.html
    ##### Function Space ######
    f_space = Scatter(title = 'Objective Space',
                        labels = obj_labels)
    f_space.add(F)
    # if pf is not None:
    #     f_space.add(pf)
    f_space.save(f'{plotDir}/obj-space.png')
    ##### Variable Space ######
    f_space = Scatter(title = 'Design Space',
                        labels = obj_labels)
    f_space.add(X)
    # if pf is not None:
    #     f_space.add(pf)
    f_space.save(f'{plotDir}/var-space.png')

    ##### Variable vs. Objective Plots ######
    # extract objectives and variables columns and plot them against each other
    for x_i, x in enumerate(X.transpose()):
        for f_i, f in enumerate(F.transpose()):
            plot = Scatter(title=f'{var_labels[x_i]} vs. {obj_labels[f_i]}',
                            labels=[var_labels[x_i], obj_labels[x_i]]
                            )
            xy = np.column_stack((x,f))
            plot.add(xy)
            plot.save(f'{plotDir}/{var_labels[x_i]}-vs-{obj_labels[f_i]}.png')

    # if there are more than 2 objectives create array of scatter plots comparing
    # the trade-off between 2 objectives at a time
    if len(F.transpose()) > 2:
        ####### Pair Wise Objective Plots #######
        # Pairwise Scatter Plots of Function Space
        plot = Scatter(tight_layout=True)
        plot.add(F, s=10)
        plot.add(F[-1], s=30, color="red")
        plot.save(f'{plotDir}/pairwise-scatter.png')
