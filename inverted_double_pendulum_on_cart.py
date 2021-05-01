import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def double_pend(y, t, K1, K2, K3, K4, g, R):

    theta_1, omega_1, theta_2, omega_2 = y
    M = np.cos(theta_1) * np.cos(theta_2) + np.sin(theta_1) * np.sin(theta_2)
    N = np.cos(theta_1) * np.sin(theta_2) - np.sin(theta_1) * np.cos(theta_2)

    a = (K1 * theta_1 + K2 * omega_1 + K3 * theta_2 + K4 * omega_2) 
    alpha = (g / R) * np.sin(theta_2) + (a / R) * np.cos(theta_2) - (omega_1 ** 2) * N
    beta = (g / R) * np.sin(theta_1) + (a / R) * np.cos(theta_1) + (1 / 2) * (omega_2 ** 2) * N

    d_omega_1_dt = (beta - (M / 2) * alpha) / (1 - (M ** 2) / 2)
    d_omega_2_dt = alpha - M * d_omega_1_dt

    dydt = [omega_1, d_omega_1_dt, omega_2, d_omega_2_dt]
    return dydt

K1 = 0
K2 = 0
K3 = 0
K4 = 0

K1 = -158.2
K2 = -18
K3 = 175.567
K4 = 35.45

y0 = [0.1, 0.0, 0.3, 0.0]

t = np.linspace(0,15,100000)

sol = odeint(double_pend, y0, t, args=(K1,K2,K3,K4,9.8,1))

plt.plot(t, sol[:, 0], 'b', label='theta_1(t)')

plt.plot(t, sol[:, 2], 'g', label='theta_2(t)')

plt.legend(loc='best')

plt.xlabel('t')

plt.grid()

plt.show()