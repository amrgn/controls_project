import numpy as np
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import gym
from gym.wrappers import Monitor
import gym_inverted_pend
import time
from matplotlib import animation
import control

#cart starting position and velocity
x0 = 0 # I'll actually shift the average position to keep the cart on the screen longer
v0 = 0

L = 0.5 #length of pendulum (in meters)

def pend(y, t, K1, K2, L = 0.5, g = 9.8):
    theta, omega = y
    dydt = [omega, g/L * np.sin(theta) - np.cos(theta) / L * (K1 * theta + K2 * omega)]
    return dydt

def pendPID(y, t, K1, K2, K3, L = 0.5, g = 9.8):
    alpha, theta, omega = y
    dydt = [theta, omega, g/L * np.sin(theta) - (np.cos(theta) / L) * (K1 * theta + K2 * omega + K3 *alpha)]
    return dydt

K1 = 50
K2 = 4
K3 = 20

y0 = [ np.pi * 0.1, 0]
y0PID = [ 0,  np.pi /3, 0]

max_time = 10

dt = 1e-3

t = np.linspace(0, max_time, int(max_time / dt))

sol = odeint(pend, y0, t, args=(K1,K2,L))
solPID = odeint(pendPID, y0PID, t, args=(K1,K2,K3,L))


def ProotlocusPlot():
    G = control.TransferFunction(L, (L, 0 , -9.8 + K1))

    rlist, klist = control.rlocus(G, grid = False)
    plt.title("Root Locus diagram, K=" + str(K1))

    plt.show()
#ProotlocusPlot()

def PDrootlocusPlot():
    G = control.TransferFunction(L, (L, K2 , -9.8 + K1))

    rlist, klist = control.rlocus(G, grid = False)
    plt.title("Root Locus diagram, K1=" + str(K1) +" and K2= " +str(K2))

    plt.show()
    

#PDrootlocusPlot()

def PIDrootlocusPlot():
    G = control.TransferFunction((L,0), (L, K2 , -9.8 + K1, K3))
    rlist, klist = control.rlocus(G, grid = False)
    plt.title("Root Locus diagram, K1=" + str(K1) +", K2= " +str(K2)+ ", and K3="+ str(K3))
    plt.show()

PIDrootlocusPlot()







def gen_plot_PD():
    
    fig, axs = plt.subplots(2)

    axs[0].plot(t, sol.T[0], 'b', label=r'$\theta(t)$')
    axs[0].plot(t, sol.T[1], 'm', label=r'$\omega(t)$')
    axs[0].legend(loc='best')
    axs[0].set_title(r'$\theta$ and $\omega$ as functions of time')
    axs[0].set(xlabel='t', ylabel='rad (or rad/s)')

    axs[1].grid()

    axs[1].plot(sol.T[0],sol.T[1])
    axs[1].set_title(r'Dynamical plot of $\omega$ against $\theta$')
    axs[1].set(xlabel=r'$\theta(t)$', ylabel=r'$\omega(t)$')

    axs[1].grid()

    plt.show()




#plt.plot(sol.T[0],sol.T[1])

#plt.show()

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


def PlotPID():
    plt.plot(t, solPID[:, 1], 'b', label='theta(t)')
    plt.plot(t, solPID[:, 0], 'g', label='alpha(t)')
    plt.plot(t, solPID[:, 2], 'r', label='omega(t)')
    plt.legend()
    plt.show()
    accel = K1 * solPID.T[1] + K2 * solPID.T[2] + K3 * solPID.T[0]

    vel = np.zeros(len(t))
    pos = np.zeros(len(t))

    vel[0] = v0
    pos[0] = x0

    dt = t[1] - t[0]
    for idx in range(1, len(t)):
        vel[idx] = vel[idx - 1] + accel[idx - 1] * dt
        pos[idx] = pos[idx - 1] + vel[idx - 1] * dt

    
    print(np.max(np.abs(pos)))

    

    #print(solPID.T[1])
    plt.plot(t, accel, 'b', label='acceleration')
    plt.plot(t, vel, 'g', label = 'velocity')
    plt.plot(t, pos, 'r', label='position')
    plt.legend()

    plt.show()
    plt.figure()
    env = gym.make('inverted-pend-v0')
    env.reset()
    xmax = max(5, np.max(np.abs(np.abs(pos))))
    xmax = min(10,xmax)
    env.set_x_max(xmax)

    #env.set_x_max(np.max(np.abs(pos))
    #pos = pos - np.average(pos)
    print(np.max(np.abs(np.abs(pos))))
    env.set_length(L)
    frames = []

    for i in range(len(solPID.T[0])):
        #frames.append(env.render(mode="rgb_array"))
        env.render()
        env.set_theta(solPID.T[1][i])
        env.set_x(pos[i])
        #print("i")
    
    env.close()

    #print(frames)
    #save_frames_as_gif(frames, filename="PID-PIover3.gif", xmax= xmax)
    
    
    
PlotPID()

def PlotPD():

    gen_plot_PD()

    accel = K1 * sol.T[0] + K2 * sol.T[1]

    vel = np.zeros(len(t))
    pos = np.zeros(len(t))

    vel[0] = v0
    pos[0] = x0

    dt = t[1] - t[0]
    for idx in range(1, len(t)):
        vel[idx] = vel[idx - 1] + accel[idx - 1] * dt
        pos[idx] = pos[idx - 1] + vel[idx - 1] * dt

    
    


    env = gym.make('inverted-pend-v0')
    env.reset()
    xmax = max(5, np.max(np.abs(np.abs(pos))))
    xmax = min(20,xmax)
    env.set_x_max(xmax)
    print(np.max(np.abs(pos)))
    #pos = pos - np.average(pos) # set average position to 0
    print(np.max(np.abs(pos)))
    env.set_length(L)
    frames =[]
    for i in range(len(sol.T[0])):
        #frames.append(env.render(mode="rgb_array"))
        env.render()
        #time.sleep(0.1)
        env.set_theta(sol.T[0][i])
        env.set_x(pos[i])
        #print("i")
    env.close()
    #save_frames_as_gif(frames, filename="PD-K1<G.gif")


#PlotPD()




