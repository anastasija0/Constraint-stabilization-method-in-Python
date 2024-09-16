import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#konstante
m = 0.5
g = 9.81
omega = 1
epsilon= 100
b=1

def system(t, variables):
    """funkcija koja se poziva tokom izvrsavanja solve_ivp funkcije,
    vraca prve izvode promenlljivih"""
    r,z,r_d,z_d=variables
    drdt = r_d
    dzdt = z_d
    dz2dt = (2*b*drdt*drdt+2*b**r*r*omega*omega-4*b*b*g*r*r-2*epsilon*dzdt+4*b*epsilon*r*drdt-epsilon*epsilon*z+epsilon*epsilon*b*r*r)/(1+4*b*b*r*r)
    dr2dt = r*omega*omega-2*b*r*dz2dt-2*b*r*g
    
    return [drdt, dzdt, dr2dt, dz2dt]

#pocenti uslovi
initial_conditions = [2.0,4.0, 0.0, 0.0]


#resavanje diferencijalnih jednacina
#dobijaju se pozicije i brzine promenljivih
time_span = (0, 10)
solution = solve_ivp(system, time_span, initial_conditions, method='RK45')
time_points = solution.t
positions = solution.y[0:2, :]
drdt_values = solution.y[2, :]
dzdt_values = solution.y[3, :]
dz2dt_values = np.gradient(dzdt_values, time_points)
dr2dt_values=np.gradient(drdt_values,time_points)


g_values = np.empty(dz2dt_values.size)
for i in range(dz2dt_values.size):
    g_values[i] = g


lambda_ = m*(dz2dt_values+g_values)


#crtanje grafika z(t) i r(t)
plt.figure(figsize=(10, 6))
plt.plot(time_points, positions[0, :], label='r')
plt.plot(time_points, positions[1, :], label='z')
plt.xlabel('vreme')
plt.ylabel('z,r')
plt.legend()
plt.title('Grafik zavisnosti z(t) i r(t)')
plt.show()

#crtanje trajektorije 
plt.figure(figsize=(10, 6))
plt.plot(positions[0, :], positions[1, :])
plt.xlabel('r')
plt.ylabel('z')
plt.legend()
plt.title('z(r) za epsilon=100')
plt.show()

#crtanje Nz komponente sile reakcije
Nz=lambda_
plt.figure(figsize=(10, 6))
plt.plot(time_points, Nz)
plt.xlabel('t')
plt.ylabel('Nz')
plt.legend()
plt.title('Grafik zavisnosti Nz(t)')
plt.show()


#crtanje Nr komponente sile reakcije
Nr=-2*b*(positions[0, :]*lambda_)
plt.figure(figsize=(10, 6))
plt.plot(time_points, Nr)
plt.xlabel('t')
plt.ylabel('Nr')
plt.legend()
plt.title('Grafik zavisnosti Nr(t)')
plt.show()

#crtanje sile ogranicenja
f=positions[1, :]-b*positions[0, :]*positions[0, :]
plt.figure(figsize=(10, 6))
plt.ylim(0, 5)
plt.plot(time_points, f)
plt.xlabel('t')
plt.ylabel('f')
plt.legend()
plt.title('Grafik zavisnosti f(t) kada je pocetan polozaj kuglice van parabole')
plt.show()