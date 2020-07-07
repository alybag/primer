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

#Plot Error: 
plt.rc('font', size=20)
plt.figure(figsize=(20,15))
plt.plot(dt, err, 'o', label='Error')
plt.title('Error')
plt.xlabel('dt')
plt.ylabel('error')
plt.legend(loc='top left')
plt.show()

#Plot 1e0





#Plot 1e-1





#Plot 1e-2





#Plot 1e-3





#Plot 1e-4





#Plot 1e-5





#Plot 1e-6
