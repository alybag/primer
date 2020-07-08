#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

#Analytical Solution
def x_exact(t):
    return np.exp((-3)*t)

#Explicit Euler
def solve_EE(dt):
    t_beg = 0
    t_end = 9
    nt = int((t_end-t_beg)/dt+1)
    
    t = np.zeros(nt)
    x = np.zeros(nt)
    
    x_0 = 1
    
    t[0] = t_beg
    x[0] = x_0
    
    for i in range(nt-1):
        x[i+1] = (1-3*dt)*x[i]
        t[i+1] = t[i] + dt
        
    return t, x

#Error
dt = np.array([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
err = np.zeros(len(dt))

for i in range(len(err)):
    t, x = solve_EE(dt[i])
    x_ex = x_exact(t)
    err[i] = np.linalg.norm(x-x_ex)

t_beg = 0
t_end = 9

#Plot Error: 
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(dt, err, 'o', label='Error')
plt.title('Error')
plt.xlabel('dt')
plt.ylabel('error')
plt.legend(loc='upper left')
plt.show()
plt.savefig('error.png')

#Plot 1e-0
output0 = np.loadtxt('/fslhome/alybag/training/data/output0.dat', delimiter = ' ')
t0 = output0[:, 0]
x0 = output0[:, 1]

t_ex = np.linspace(t_beg, t_end, 101)
x_ex = np.exp((-3)*t_ex)

plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t0, x0, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-0')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-0.png')





#Plot 1e-1





#Plot 1e-2





#Plot 1e-3





#Plot 1e-4





#Plot 1e-5





#Plot 1e-6
