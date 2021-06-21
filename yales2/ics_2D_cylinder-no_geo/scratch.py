# import os
# import scipy.integrate as integrate
# from scipy.integrate import quad
# import numpy as np
#
# import matplotlib.pyplot as plt
#
# omega = 3
# freq = 1
#
# ######## Objective 2: Position of vortex #########
# # Objective 2: Power consumed by rotating cylinder
# D = 1  # [m] cylinder diameter
# t = 0.1  # [m] thickness of cylinder wall
# r_o = D/2  # [m] outer radius
# r_i = r_o-t  # [m] inner radius
# d = 2700  # [kg/m^3] density of aluminum
# L = 1  # [m] length of cylindrical tube
# V = L*np.pi*(r_o**2-r_i**2) # [m^3] volume of cylinder
# m = d*V # [kg] mass of cylinder
# I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
# P_cyc = 0.5*I*quad(lambda t : (omega*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle
# print(P_cyc)
#
# n_pts = 100
# t = np.linspace(0, 2*np.pi, n_pts)
# amp = omega*np.sin(t)
# plt.plot(t, amp)
# plt.show()



## KWARGS
# def fun(**kwargs):
#     return kwargs
#
# c = 'c'
# kwargs = fun(kw_a='a', kw_b='b')
#
# jobFile = kwargs.get('kw_c', c)
# print(f'{jobFile}')




# ## LISTS
# import numpy as np
# a = [2, 2]
# b = [1, 1]
#
# print(np.subtract(b, a))
#
# print(np.subtract(a, b) / np.subtract(a,b))


## strip string

dir = 'gen0//'
print(dir.lstrip('./'))


info = {'age':'21', 'city':'Pune'}

print(f"{info['city']}")

import numpy as np

help(np.ndarray)
