import matplotlib.pyplot as plt
import numpy as np

npts = 200
t = np.linspace(0,1,npts)
u_mean = np.full(npts, 1)
# u_fluc = 0.5*np.random.rand(npts)+0.75
denom = 4
u_fluc = np.random.uniform(u_mean-u_mean/denom, u_mean+u_mean/denom, npts)

print(len(t))
# print(len(u_mean))
print(len(u_fluc))

fig, ax = plt.subplots()

ax.plot(t, u_fluc)
ax.plot(t, u_mean)
ax.legend(( "u'", "$\\bar{u}$"))

ax.set(xlabel='Time (s)', ylabel='Flow Velocity (m/s)',
       title='Mean Velocity vs. Velocity Fluctuations')
ax.set_ylim((0,2))

fig.savefig("mean-vs-fluc.png")
# plt.show()


################################################################################

npts = 200
d = np.linspace(0,1,npts)
u_mean = d/(d+0.05)
max_fluc = 0.05
u_fluc = np.random.uniform(u_mean-max_fluc, u_mean+max_fluc, npts)

fig, ax = plt.subplots()

ax.plot(u_fluc, d)
ax.plot(u_mean, d)
ax.legend(( "u'", "$\\bar{u}$"))

ax.set(xlabel='Flow Velocity (m/s)', ylabel='Distance from Wall (m)',
       title='Mean Velocity vs. Velocity Fluctuations')
ax.set_xlim((0, 1.5))

fig.savefig("mean-vs-fluc-wall.png")
plt.show()
