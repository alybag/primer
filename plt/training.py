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
t_ex = np.linspace(t_beg, t_end, 101)
x_ex = np.exp((-3)*t_ex)

output0 = np.loadtxt('/fslhome/alybag/training/data/output0.dat', delimiter = ' ')
t0 = output0[:, 0]
x0 = output0[:, 1]

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
output1 = np.loadtxt('/fslhome/alybag/training/data/output1.dat', delimiter = ' ')
t1 = output1[:, 0]
x1 = output1[:, 1]     
                                                                                                                                 
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t1, x1, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-1')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-1.png')

#Plot 1e-2
output2 = np.loadtxt('/fslhome/alybag/training/data/output2.dat', delimiter = ' ')
t2 = output2[:, 0]
x2 = output2[:, 1]

plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t2, x2, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-2')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-2.png')

#Plot 1e-3
output3 = np.loadtxt('/fslhome/alybag/training/data/output3.dat', delimiter = ' ')
t3 = output3[:, 0]
x3 = output3[:, 1]

plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t3, x3, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-3')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-3.png')

#Plot 1e-4
output4 = np.loadtxt('/fslhome/alybag/training/data/output4.dat', delimiter = ' ')
t4 = output4[:, 0]
x4 = output4[:, 1]
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t4, x4, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-4')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-4.png')

#Plot 1e-5
output5 = np.loadtxt('/fslhome/alybag/training/data/output5.dat', delimiter = ' ')
t5 = output5[:, 0]
x5 = output5[:, 1]
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t5, x5, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-5')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-5.png')

#Plot 1e-6
output6 = np.loadtxt('/fslhome/alybag/training/data/output6.dat', delimiter = ' ')
t6 = output6[:, 0]
x6 = output6[:, 1]
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(t6, x6, 'bo', markersize=5, label='Explicit Euler')
plt.plot(t_ex, x_ex, 'r-', label='Exact Solution')
plt.title('Explicit Euler, dt = 1e-6')
plt.xlabel('t')
plt.ylabel('x')
plt.legend(loc='upper right')
plt.show()
plt.savefig('1e-6.png')
