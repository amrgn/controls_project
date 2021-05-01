import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pend(y, t, K1, K2, L = 0.5, g = 9.8):
    theta, omega = y
    dydt = [omega, g/L * np.sin(theta) - np.cos(theta) / L * (K1 * theta + K2 * omega)]
    return dydt

K1 = 20
K2 = 1

y0 = [0.1, 0.0]

t = np.linspace(0,10,100)

sol = odeint(pend, y0, t, args=(K1,K2))

plt.plot(t, sol[:, 0], 'b', label='theta(t)')

plt.plot(t, sol[:, 1], 'g', label='omega(t)')

plt.legend(loc='best')

plt.xlabel('t')

plt.grid()

plt.show()