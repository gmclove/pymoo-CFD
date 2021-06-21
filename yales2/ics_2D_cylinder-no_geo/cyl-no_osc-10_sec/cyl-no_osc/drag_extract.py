import numpy as np

dataDir = 'ics_temporals.txt'
data = np.genfromtxt(dataDir, skip_header=1)
# collect data after 100 seconds
noffset = 8 * data.shape[0] // 10
# extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
p_over_rho_intgrl_1 = data[noffset:, 4]
tau_intgrl_1 = data[noffset:, 6]
drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
print(drag)

# # max possible power consumption
# omega = 3
# freq = 1
# ######## Objective 2: Position of vortex #########
# # Objective 2: Power consumed by rotating cylinder
# D = 1  # [m] cylinder diameter
# t = 0.1  # [m] thickness of cylinder wall
# r_o = D/2  # [m] outer radius
# r_i = r_o-t  # [m] inner radius
# d = 2700  # [kg/m^3] density of aluminum
# L = 1  # [m] length of cylindrical tube
# V = L*np.pi*(r_o**2-r_i**2)
# m = d*V
# I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
# E = 0.5*I*omega**2  # [J] or [(kg m^2)/s^2] energy consumption at peak rotational velocity (omega)
# P_avg = E*4*freq  # [J/s] average power over 1/4 cycle
# print(P_avg)
