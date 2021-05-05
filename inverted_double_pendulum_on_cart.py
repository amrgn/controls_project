import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import gym
import gym_inverted_pend

g = 9.8
R = 5

#cart starting position and velocity
x0 = 0
v0 = 0

def double_pend(y, t, K1, K2, K3, K4, g, R):

    theta_1, omega_1, theta_2, omega_2 = y
    M = np.cos(theta_1) * np.cos(theta_2) + np.sin(theta_1) * np.sin(theta_2)
    N = np.cos(theta_1) * np.sin(theta_2) - np.sin(theta_1) * np.cos(theta_2)

    a = -(K1 * theta_1 + K2 * omega_1 + K3 * theta_2 + K4 * omega_2)
    #if t < 0.1:
    #    a += 50
    alpha = (g / R) * np.sin(theta_2) - (a / R) * np.cos(theta_2) - (omega_1 ** 2) * N
    beta = (g / R) * np.sin(theta_1) - (a / R) * np.cos(theta_1) + (1 / 2) * (omega_2 ** 2) * N

    d_omega_1_dt = (beta - (M / 2) * alpha) / (1 - (M ** 2) / 2)
    d_omega_2_dt = alpha - M * d_omega_1_dt

    dydt = [omega_1, d_omega_1_dt, omega_2, d_omega_2_dt]
    return dydt

# desired roots of the polynomial in the denominator of the response function. COMPLEX ROOTS MUST OCCUR IN CONJUGATE PAIRS
# s^4 + a_3 s^3 + a_2 s^2 + a_1 s + a_0 = (s - alpha_1)(s - alpha_2)(s - alpha_3)(s - alpha_4)

alpha_1 = -1
alpha_2 = -2
alpha_3 = -2.5
alpha_4 = -3

#a_0 through a_3 obtained by Vieta's formula
a_0 = alpha_1 * alpha_2 * alpha_3 * alpha_4
a_1 = -(alpha_1 * alpha_2 * alpha_3 + alpha_1 * alpha_2 * alpha_4 + alpha_1 * alpha_3 * alpha_4 + alpha_2 * alpha_3 * alpha_4)
a_2 = (alpha_1 * alpha_2 + alpha_1 * alpha_3 + alpha_1 * alpha_4 + alpha_2 * alpha_3 + alpha_2 * alpha_4 + alpha_3 * alpha_4)
a_3 = -(alpha_1 + alpha_2 + alpha_3 + alpha_4)

# The denominator of Q1, Q2 is R^2(s^4 - (K2 / R) * s^3 - ((4g + K1) / R)* s^2 + (2g / R^2) * (K2 + K4) * s + (2g / R^2) * (K1 + K3 + 1))

# solving for K1, K2, K3, K4 gives

K1 = np.real(-a_2 * R - 4 * g)
K2 = np.real(-a_3 * R)
K3 = np.real(a_0 * (R ** 2) / (2 * g) - K1 - 1)
K4 = np.real(a_1 * (R ** 2) / (2 * g) - K2)

print('K1 = ', K1, 'K2 = ', K2, 'K3 = ', K3, 'K4 = ', K4)


#K1 = 0
#K2 = 0
#K3 = 0
#K4 = 0

# note that the double pendulum is especially sensitive to initial conditions,
# and the nonlinear effects can take over fast, causing a bad control model.

y0 = [0, 0, 0.15 * np.pi, 0]

max_time = 10

dt = 1e-3

t = np.linspace(0, max_time, int(max_time / dt))


sol = odeint(double_pend, y0, t, args=(K1,K2,K3,K4,g,R))

# integrate acceleration to find velocity and position of cart
accel = -(K1 * sol.T[0] + K2 * sol.T[1] + K3 * sol.T[2] + K4 * sol.T[3])
vel = np.zeros(len(t))
pos = np.zeros(len(t))

vel[0] = v0
pos[0] = x0

dt = t[1] - t[0]

for idx in range(1, len(t)):
    vel[idx] = vel[idx - 1] + accel[idx - 1] * dt
    pos[idx] = pos[idx - 1] + vel[idx - 1] * dt



fig, axs = plt.subplots(2)

axs[0].plot(t, sol.T[0], 'b', label=r'$\theta_1(t)$')
axs[0].plot(t, sol.T[1], 'm', label=r'$\omega_1(t)$')

axs[0].plot(t, sol.T[2], 'g', label=r'$\theta_2(t)$')
axs[0].plot(t, sol.T[3], 'r', label=r'$\omega_2(t)$')

axs[0].legend(loc='best')
axs[0].set_title(r'$\theta$ and $\omega$ as functions of time')
axs[0].set(xlabel='t', ylabel='rad (or rad/s)')

axs[1].grid()

axs[1].plot(sol.T[0],sol.T[1],label=r'$\omega_1(t)$ against $\theta_1(t)$')
axs[1].plot(sol.T[2],sol.T[3],label=r'$\omega_2(t)$ against $\theta_2(t)$')
axs[1].set_title(r'Dynamical plot of $\omega$ against $\theta$')
axs[1].set(xlabel=r'$\theta_1(t)$ or $\theta_2(t)$', ylabel=r'$\omega_1(t)$ or $\omega_2(t)$')
axs[1].legend(loc='best')

axs[1].grid()

plt.show()

def save_frames_as_gif(frames, path='./', filename='gym.gif', xmax= 5):

    #Mess with this to change frame size
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")
    #plt.xticks(np.linspace(-xmax, xmax, 600))
    #x.set(xlim = (-xmax,xmax), ylim = None )

    def animate(i):
        patch.set_data(frames[18*i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = int(len(frames)/18), interval=1)
    
    #print(1/dt)

    anim.save(path + filename, writer='Pillow', fps=1000)




env = gym.make('double-inverted-pend-v0')
env.reset()
frames =[]
for i in range(len(sol.T[0])):
    frames.append(env.render(mode="rgb_array"))
    #time.sleep(0.1)
    env.set_theta(sol.T[0][i])
    env.set_theta2(sol.T[2][i])
    env.set_x(pos[i])
    #print("i")
env.close()
env.render()

#save_frames_as_gif(frames, filename="PD-K1<G.gif")