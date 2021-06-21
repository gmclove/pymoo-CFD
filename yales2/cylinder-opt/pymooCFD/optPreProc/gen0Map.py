import os
import numpy as np

def gen0Map():
    # checkpointDir = 'checkpoint' + '.npy'
    checkpointDir = 'checkpoint.npy'
    plotDir = './plots'
    ########################################################################################################################
    checkpoint, = np.load(checkpointDir, allow_pickle=True).flatten()
    algorithm = checkpoint

    # X = algorithm.pop.get('X')
    # F = algorithm.pop.get('F')
    X = algorithm.callback.data['var']
    F = algorithm.callback.data['obj']
    print(X)
    print(F)
    ########################################################################################################################
    ##### SCATTER PLOTS #######
    ###########################
    from pymoo.visualization.scatter import Scatter
    #
    # for x in range(len(X)):
    #     for f in range(len(F)):
    #         var = X[x]
    #         obj = F[f]
    #         plot = Scatter(title=f'{var_labels[x]}={var} {obj_labels[f]}={obj}', labels=[var, obj])
    #         plot.add()
