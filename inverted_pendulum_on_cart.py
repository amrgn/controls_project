import numpy as np
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import gym
import gym_inverted_pend
import time

def pend(y, t, K1, K2, L = 0.5, g = 9.8):
    theta, omega = y
    dydt = [omega, g/L * np.sin(theta) - np.cos(theta) / L * (K1 * theta + K2 * omega)]
    return dydt

K1 = 50
K2 = 3

y0 = [np.pi * 0.45, 0]

t = np.linspace(0,10,10000)

sol = odeint(pend, y0, t, args=(K1,K2))

plt.plot(t, sol[:, 0], 'b', label='theta(t)')
#
#plt.plot(t, sol[:, 1], 'g', label='omega(t)')
#
#plt.legend(loc='best')
#
#plt.xlabel('t')
#
#plt.grid()
#
#plt.show()

plt.figure()

acceleration = K1 * sol.T[0] + K2 * sol.T[1]


velocity = cumtrapz(acceleration, dx = 0.0001)
pos = cumtrapz(velocity, dx = 1/1000)

print (pos)
#plt.plot(sol.T[0],sol.T[1])

#plt.show()

#print(sol.T[0])

env = gym.make('inverted-pend-v0')
env.reset()
for i in range(len(sol.T[0])):
    env.render()
    #time.sleep(0.1)
    env.set_theta(sol.T[0][i])
    env.set_x(pos[i])
    #print("i")
env.close()
env.render()