from pymooIN import *

algorithm.next()

print(f'Generation {algorithm.n_gen} complete')

np.save("checkpoint", algorithm)
