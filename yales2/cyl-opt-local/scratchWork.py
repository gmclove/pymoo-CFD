### numpy loadtxt/savetxt

# import numpy as np

# a = [[1,2],[3,4],[4,5]]
#
# np.savetxt('test.txt',a)
#
# b = [[5,7],[8,9],[10,11]]
#
# with open('test.txt', 'a+') as file:
#     file.write('\n')
#     np.savetxt(file, b)
#
# print(np.loadtxt('test.txt'))


# from pymooCFD.setupOpt import loadCP
# alg = loadCP(checkpointFile = 'checkpoint.npy')
# print(alg.n_gen)
# print(alg.callback.data['var'])
