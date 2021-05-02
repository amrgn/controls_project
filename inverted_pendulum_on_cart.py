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

def pendPID(y, t, K1, K2, K3, L = 0.5, g = 9.8):
    alpha, theta, omega = y
    dydt = [theta, omega, g/L * np.sin(theta) - np.cos(theta) / L * (K1 * theta + K2 * omega + K3 *alpha)]
    return dydt

K1 = 50
K2 = 2
K3 = 30

y0 = [ np.pi * 0.45, 0]
y0PID = [ 0,  np.pi * 0.45, 0]

t = np.linspace(0,10,10000)

sol = odeint(pend, y0, t, args=(K1,K2))
solPID = odeint(pendPID, y0PID, t, args=(K1,K2,K3))









#plt.plot(sol.T[0],sol.T[1])

#plt.show()


def PlotPID():
    #plt.plot(t, solPID[:, 1], 'b', label='theta(t)')
    #plt.plot(t, solPID[:, 0], 'b', label='alpha(t)')
    #plt.plot(t, solPID[:, 2], 'b', label='omega(t)')
    acceleration = K1 * solPID.T[1] + K2 * solPID.T[2] + K3 * solPID.T[0]

    velocity = cumtrapz(acceleration, dx = 0.0001)
    pos = cumtrapz(velocity, dx = 1/1000)
    print(solPID.T[1])
    plt.plot(t[:9998], acceleration[:9998], 'b', label='acceleration')
    plt.plot(t[:9998], velocity[:9998], 'g', label = 'velocity')
    plt.plot(t[:9998], pos, 'r', label='position')
    plt.legend()

    plt.show()
    plt.figure()
    env = gym.make('inverted-pend-v0')
    env.reset()
    for i in range(len(solPID.T[0])):
        env.render()
        #time.sleep(0.1)
        env.set_theta(solPID.T[1][i])
        env.set_x(pos[i])
        #print("i")
    env.close()
    env.render()
    
PlotPID()

def PlotPD():
    plt.plot(t, sol[:, 0], 'g', label='theta(t)')
    plt.plot(t, sol[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.figure()

    acceleration = K1 * sol.T[0] + K2 * sol.T[1]
    velocity = cumtrapz(acceleration, dx = 0.0001)
    pos = cumtrapz(velocity, dx = 1/1000)



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

#PlotPD()