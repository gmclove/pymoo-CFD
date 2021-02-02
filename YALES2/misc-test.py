import numpy as np

def outer():
    def inner():
        print('inner')
        print(x)

    x = 1
    inner()

outer()