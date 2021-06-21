import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

print(os.getcwd())

def joukowski_map(mu_x, mu_y, num_pt):
    # center of circle in complex plane
    comp_cent = np.array([mu_x, mu_y])
    # radius of circle in complex plane /
    # distance from center to point (-1,0) in complex plane
    r = np.sqrt((comp_cent[0]-1)**2 + (comp_cent[1]-0)**2)

    # Circle coordinates calculations
    angle = np.linspace(0, 2*np.pi, num_pt) # 500 points along circle [0, 2*pi]
    comp_r = comp_cent[0] + r*np.cos(angle) # real coordinates along circle (horz.)
    comp_i = comp_cent[1] + r*np.sin(angle) # imaginary coordinates along circle (vert.)

    # Cartesian components of the Joukowsky transform
    x = ((comp_r)*(comp_r**2+comp_i**2+1))/(comp_r**2+comp_i**2)
    y = ((comp_i)*(comp_r**2+comp_i**2-1))/(comp_r**2+comp_i**2)
    plt.plot(x,y)
    plt.show()

    ########################################
    # change chord length to be from x=0 to 1
    # Compute the scale factor (actual chord length)
    c = np.max(x)-np.min(x)
    # Leading edge current position
    LE = np.min(x/c)
    # Corrected position of the coordinates
    x = x/c-LE # move the leading edge
    y = y/c

    # return 500 points that make up airfoil shape
    return x, y

def af_area(x, y):
    '''
    Use composite simpson's rule to find the approximate area inside the airfoil.

    scipy.integrate.simpson(y[, x, dx, axis, even])
	Integrate y(x) using samples along the given axis and the composite Simpson’s rule.
    '''
    area = integrate.simpson(y, x)


    # # Once the different lines are computed, the area will be computed as the integral of those lines
    #
    # # In case the lower surface of the airfoil interceps the y = 0 axis, it must be divided so all areas
    # # are computed independently
    # lowerNeg = lower[lower[:,1]<0,:]
    # lowerPos = lower[lower[:,1]>0,:]
    #
    # # Upper surface area
    # A1 = integrate.simps(upper[np.argsort(upper[:,0]),1], upper[np.argsort(upper[:,0]),0])
    # # Lower surface area for points with negative y
    # A2 = -integrate.simps(lowerNeg[np.argsort(lowerNeg[:,0]),1], lowerNeg[np.argsort(lowerNeg[:,0]),0])
    # # Possible lower surface area for points with positive y
    # A3 = integrate.simps(lowerPos[np.argsort(lowerPos[:,0]),1], lowerPos[np.argsort(lowerPos[:,0]),0])
    #
    # # The area will be the sum of the areas and substracting the possible intercept of both
    # area = A1 + A2 - A3


# example data:
# -0.2125,0.084375,-11.441091805932968,-0.1382712420131816
x, y = joukowski_map(-0.2125, 0.084375, 500)
area = abs(simpson(y, x))
print(area)
