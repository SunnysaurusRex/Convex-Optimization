# SZ file_2 (phe, tyr)
import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm

# ODE parameters
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

# Additional parameters that involve other amino acids; expected to inhibit cell growth
K_arg = 2.5e-2
K_asp = 1e-1
K_his = 1e-2
K_ile = 5e-2
K_leu = 1.5e-2
K_lys = 2e-2
K_phe = 5e-2
K_ser = 2.5e-2
K_thr = 5e-2
K_val = 2.5e-2
Y_ala_x = 8e-12
Y_arg_asp = 1e-2
Y_arg_glu = 1e-2
Y_arg_pro = 1e-2
Y_asn_asp = 1.8e-1 
Y_asp_arg = 1e-2
Y_asp_x = 1e-15
Y_cys_ser = 1 
Y_glu_arg = 1e-3
Y_glu_gln = 2e-1
Y_glu_his = 1
Y_glu_pro = 1e-2
Y_gly_ser = 4e-1
Y_glu_x = 2e-16
Y_lac_glc = 1.2
Y_lys_x = 2e-14
Y_pro_arg = 1.2
Y_pro_glu = 1
Y_ser_gly = 1e-1
Y_tyr_phe = 1
Y_x_ala = 1.5e10
Y_x_arg = 4.6e9
Y_x_asp = 2.5e9
Y_x_cys = 2e9
Y_x_glu = 6.5e9
Y_x_his = 2.4e10
Y_x_ile = 6.85e9
Y_x_leu = 2.5e9
Y_x_lys = 6.3e9
Y_x_met = 1.9e10
Y_x_phe = 1.4e10
Y_x_pro = 1e11
Y_x_ser = 1.3e9
Y_x_thr = 1e10
Y_x_tyr = 4.3e9
Y_x_val = 6e9

# get the grid bounds for each ODE parameter in [=] g/L -> mM
molwt = [180.156, 1, 89.1, 174.2, 132.1, 133.1, 121.1, 147.1, 75.1, 155.2, 131.2, 131.2, 146.2,
		 149.2, 165.2, 115.1, 105.1, 119.1, 181.2, 117.1] 	# vector of molecular weights
# bounds of each thing in g/L
glc = [1, 10]
gln = [1, 4]
his = [.015, .152]
phe = [.015, .313]
tyr = [.029, .197]
# make vector of bounds in g/L
hyperparams = [glc, gln, his]
# convert each parameter bound to mM
#for i in range(len(hyperparams)):
#	hyperparams[i] = np.array(hyperparams[i])/molwt[i]*1000

# cell density initial conditions
X0 = 1e6			# inoculum, cell density [=] cells/L
Xv0 = X0*.95		# Assume 95% viability
Xd0 = X0-Xv0 		# assume 5% dead cells or 95% viable cells
t1 = 150			# solve ODEs until 150 hours
GLC0 = 50
GLN0 = 4
amm = 1e-10 	# ammonia initial condition, !0 but =~ 0 because it appears in a denom. somewhr

IC = np.array([Xv0, Xd0, GLC0, GLN0, amm])
# ODE function
def cells2(t, w): # inputs: time value, solution vector at current iteration
	# decompose w vector
	Xv = w[0]
	Xd = w[1]
	GLC = w[2]
	GLN = w[3]
	AMM = w[4]
	HIS = w[5]

	# specfic growth and death rates
	mu = mu_min + (mu_max-mu_min)*GLC*GLN*HIS/(K_glc+GLC)/(K_gln+GLN)/(K_his+HIS)
	mu_d = mu_d_max/(1+(K_d_amm/AMM)**d_n)
	# intermediate variables for ODEs
	# these are independent
	M_gln = a1*GLN/(a2+GLN)
	Q_his = -mu/Y_x_his 	# d(HIS)/dt is independent and affects \mu

	Q_gln = -mu/Y_x_gln - M_gln
	Q_glc = -mu/Y_x_glc - M_glc 
	Q_amm = -Y_amm_gln*Q_gln

	# ODE
	dXv = (mu-mu_d)*Xv
	dXd = (mu_d)*Xd
	dGLN = Q_gln*Xv - K_d_gln*GLN
	dGLC = Q_glc*Xv
	dAMM = Q_amm*Xv + K_d_gln*GLN
	dHIS = Q_his*Xv

	return np.array([dXv, dXd, dGLN, dGLC, dAMM, dHIS])

N = 20
his = np.array(his)/155.2*1000
print(his)
his_x = np.linspace(his[0], his[1], N)

def objective_his(x):
	# make initial condition vector to feed into ODE system to solve for cell density
	ICn = np.append(IC, x)
	Xv = integrate.solve_ivp(cells2, (0, t1), ICn, rtol=1e-6) # solve ODEs
	Xv_max = np.max(Xv.y[0]) 	# max cell density
	# find where it is at the max--for what t
	index = np.where(Xv.y[0] == Xv_max)
	t_max = Xv.t[index] 	# t_{max}
	if t_max < 1:
		t_max = 1
	obj = -Xv_max/t_max 	# this is the objective function, return it
	plt.scatter(x, obj)

	return obj

#for x in his_x:
#	objective_his(x)


# Do a 2D grid search for PHE and TYR
phe = np.array(phe)/165.2*1000
tyr = np.array(tyr)/181.2*1000

print('phe', phe)
print('tyr', tyr)

N = 20
phe_grid = np.linspace(phe[0], phe[1], N)
tyr_grid = np.linspace(tyr[0], tyr[1], N)
IC = np.array([Xv0, Xd0, GLC0, GLN0, amm, his[1]])
def cells3(t, w): # inputs: time value, solution vector at current iteration
	# decompose w vector
	Xv, Xd, GLC, GLN, AMM, HIS, PHE, TYR = w
	# specfic growth and death rates
	mu = mu_min + (mu_max-mu_min)*GLC*GLN*HIS*PHE/(K_glc+GLC)/(K_gln+GLN)/(K_his+HIS)/(K_phe+PHE)
	mu_d = mu_d_max/(1+(K_d_amm/AMM)**d_n)
	# intermediate variables for ODEs
	# these are independent
	M_gln = a1*GLN/(a2+GLN)
	Q_gln = -mu/Y_x_gln - M_gln
	Q_glc = -mu/Y_x_glc - M_glc 
	Q_amm = -Y_amm_gln*Q_gln
	Q_his = -mu/Y_x_his

	Q_phe = -mu/Y_x_phe 	# PHE is in the mu expression
	Q_tyr = -mu/Y_x_tyr - Y_tyr_phe*Q_phe		# TYR not in the mu expression but consumes PHE

	# ODE
	dXv = (mu-mu_d)*Xv
	dXd = (mu_d)*Xd
	dGLN = Q_gln*Xv - K_d_gln*GLN
	dGLC = Q_glc*Xv
	dAMM = Q_amm*Xv + K_d_gln*GLN
	dHIS = Q_his*Xv
	dPHE = Q_phe*Xv
	dTYR = Q_tyr*Xv

	return np.array([dXv, dXd, dGLN, dGLC, dAMM, dHIS, dPHE, dTYR])

def objective_phetyr(x):
	# decompose the grid search params
	phe, tyr = x
	# make initial condition vector to feed into ODE system to solve for cell density
	ICn = np.append(IC, x)
	Xv = integrate.solve_ivp(cells3, (0, t1), ICn, rtol=1e-7) # solve ODEs
	Xv_max = np.max(Xv.y[0]) 	# max cell density
	# find where it is at the max--for what t
	index = np.where(Xv.y[0] == Xv_max)
	t_max = Xv.t[index] 	# t_{max}
	if t_max < 1:
		t_max = 1
	obj = -Xv_max/t_max 	# this is the objective function, return it
	return obj

phe_tyr = np.empty([N,N])
for i in range(N):
	for j in range(N):
		zeta = objective_phetyr([phe_grid[i], tyr_grid[j]])
		phe_tyr[i][j] = zeta
		print(i, j, zeta)

phe_grid, tyr_grid = np.meshgrid(phe_grid, tyr_grid, indexing='ij')

plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(phe_grid, tyr_grid, phe_tyr, cmap=cm.coolwarm)
ax.view_init(20, 70)
ax.set_xlabel('phe')
ax.set_ylabel('tyr')

minimum_obj = np.min(phe_tyr)
index_min = np.where(phe_tyr == minimum_obj)
print(index_min)
print(phe_tyr[index_min])
#plt.xlabel('time / hours')
#plt.ylabel('cell density / $cells\\cdot mL^{-1}$')
#plt.title('Viable cell density solution at optimal parameters curve')

plt.show()