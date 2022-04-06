# SZ 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# parameters
# specific growth rate, \mu
mu_min = 2e-2	# minimum specific growth rate [=] 1/h
mu_max = 6.5e-2	# maximum specific growth rate [=] 1/h
K_glc = 3e-2	# Monod constant for glucose [=] mM
K_gln = 3e-3	# " " " glutamine [=] mM
# specific death rate, \mu_{d}
mu_d_max = 1e-2	# maximum specfic death rate [=] 1/h
K_d_amm = 15	# constant for cell death due to ammonia accumulation [=] mM
d_n = 2.3 		# the exponent in the equation
# GLN glutamine differential equation
K_d_gln = 9e-3	# constant for glutamine degradation [=] 1/h
Y_x_gln = 4e8 	# yield of cells on glutamine [=] cells/mmol
Y_gln_glu = 1 	# yield of glutamine from glutamate [=] mmol/mmol
a1 = 3.4e-13	# alpha_1 of glutamine maintenance coefficient [=] mM/L/cell/h
a2 = 4 			# alpha_2 " " " " [=] mM
# GLC glucose differential equation
Y_x_glc = 1.4e8 # yield of cells on glucose [=] cells/mmol
M_glc = 4.8e-14	# maintenance coefficient for glucose [=] mmol/cell/h
# AMM ammonia differential equation
Y_amm_gln = 9e-1 # yield of ammonia on glutamine [=] mmol/mmol
# X_d dead cells differential equation
K_lysis = 1.3e-2 # Monod constant for lyssis [=] mM
# initial conditions
Xv0 = 1e6/.1		# inoculum [=] cells/L ; assume 100% viability <=> X_v
Xd0 = Xv0/.95 		# assume 5% dead cells or 95% viable cells
GLC_0 = 25		# glucose concentration [=] mM
GLN_0 = 4 		# glutamine concentration [=] mM
AMM_0 = 1e-5	# ammonia concentration [=] mM 

t = 90 		# solve ODEs up to this time in hours

IC = np.array([Xv0, Xd0, GLC_0, GLN_0, AMM_0])

# ODE function
def HEK293(t, w): # inputs: time value, solution vector at current iteration
	# decompose w vector
	Xv = w[0]
	Xd = w[1]
	GLC = w[2]
	GLN = w[3]
	AMM = w[4]
	# specfic growth and death rates
	mu = mu_min + (mu_max-mu_min)*GLC*GLN/(K_glc+GLC)/(K_gln+GLN)
	mu_d = mu_d_max/(1+(K_d_amm/AMM)**d_n)
	# intermediate variables for ODEs
	M_gln = a1*GLN/(a2+GLN)
	Q_gln = -mu/Y_x_gln - M_gln
	Q_glc = -mu/Y_x_glc - M_glc
	Q_amm = -Y_amm_gln*Q_gln
	# ODE
	dXv = (mu-mu_d)*Xv
	dXd = (mu_d-K_lysis)*Xd
	dXd = (mu_d)*Xd
	dGLN = Q_gln*Xv - K_d_gln*GLN
	dGLC = Q_glc*Xv
	dAMM = Q_amm*Xv + K_d_gln*GLN 
	return np.array([dXv, dXd, dGLN, dGLC, dAMM])

cells = integrate.solve_ivp(HEK293, (0,t), IC, rtol=1e-9)
time = cells.t
viable = cells.y[0]
dead = cells.y[1]
glutamine = cells.y[2]
glucose = cells.y[3]
nh3 = cells.y[4]
print('viability:', viable[-1]/(viable[-1]+dead[-1]))
print('mult', viable[-1]/Xv0)
print(np.max(viable), 'vialble')
print(dead)

# nondimensionalize these variables, normalize
viable = viable/np.max(viable)
dead = dead/np.max(dead)
glutamine = glutamine/np.max(glutamine)
glucose = glucose/np.max(glucose)
nh3 = nh3/np.max(nh3)
print(dead)

plt.plot(time, viable, label='Xv')
plt.plot(time, dead, label='Xd')
plt.plot(time, glutamine, label='GLN')
plt.plot(time, glucose, label='GLC')
plt.plot(time, nh3, label='AMM')

plt.title('Simplified kinetic model')
plt.xlabel('time / hours')
plt.ylabel('dimensionless value')

plt.legend()
plt.show()

h_24index = np.where(abs(time-24) < 1)
print(time[h_24index])
print(cells.y[0][h_24index])
