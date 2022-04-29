# SZ convex (asp, arg, pro, glu)
import numpy as np
import optuna
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt

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

# Additional parameters
K_arg = 2.5e-2
K_asp = 1e-1
K_his = 1e-2
K_phe = 5e-2
Y_arg_asp = 1e-2
Y_arg_glu = 1e-2
Y_arg_pro = 1e-2
Y_asp_arg = 1e-2
Y_asp_x = 1e-15
Y_glu_arg = 1e-3
Y_glu_gln = 2e-1
Y_glu_his = 1
Y_glu_pro = 1e-2
Y_glu_x = 2e-16
Y_pro_arg = 1.2
Y_pro_glu = 1
Y_tyr_phe = 1
Y_x_ala = 1.5e10
Y_x_arg = 4.6e9
Y_x_asp = 2.5e9
Y_x_glu = 6.5e9
Y_x_his = 2.4e10
Y_x_phe = 1.4e10
Y_x_pro = 1e11
Y_x_tyr = 4.3e9

# get the grid bounds for each ODE parameter in [=] g/L -> mM
molwt = [155.2, 165.2, 181.2, 133.1, 174.2, 115.1, 147.1] 	# vector of molecular weights
# bounds of each thing in g/L,
his = [.015, .152]
phe = [.015, .313]
tyr = [.029, .197]
arg = [.084, 1.331]
asp = [.013, .456]
pro = [0, .121]
glu = [.011, .642]
# make vector of bounds in g/L

# making initial condition vectors
glc0 = 55
gln0 = 4
amm0 = 1e-10 	# ammonia initial condition, !0 but =~ 0 because it appears in a denom. somewhr

hyperparams = [his, phe, tyr, asp, arg, pro, glu]
# convert each parameter bound to mM
for i in range(len(hyperparams)):
	hyperparams[i] = np.array(hyperparams[i])/molwt[i]*1000
his0 = hyperparams[0][1] 	# know previously the optimal parameter for his, phe, tyr
phe0 = hyperparams[1][1]
tyr0 = hyperparams[2][0]
hyperparams = hyperparams[3:] # the decision variables are arg, asp, pro, glu

# cell density initial conditions
X0 = 1e6			# inoculum, cell density [=] cells/L
Xv0 = X0*.95		# Assume 95% viability
Xd0 = X0-Xv0 		# assume 5% dead cells or 95% viable cells
t1 = 150			# solve ODEs until 150 hours
IC = np.array([Xv0, Xd0, glc0, gln0, amm0, his0, phe0, tyr0])

# ODE function
def cells4(t, w): # inputs: time value, solution vector at current iteration
	# decompose w vector
	Xv, Xd, GLC, GLN, AMM, HIS, PHE, TYR, ASP, ARG, PRO, GLU = w
	# specfic growth and death rates
	mu = mu_min + (mu_max-mu_min)*GLC*GLN/(K_glc+GLC)/(K_gln+GLN) \
		* ASP*ARG/(K_asp+ASP)/(K_arg+ARG) \
		* HIS*PHE/(K_his+HIS)/(K_phe+PHE)
	mu_d = mu_d_max/(1+(K_d_amm/AMM)**d_n)
	# intermediate variables for ODEs
	# these are independent
	Q_glc = -mu/Y_x_glc - M_glc
	M_gln = a1*GLN/(a2+GLN)
	Q_his = -mu/Y_x_his
	Q_phe = -mu/Y_x_phe
	Q_tyr = -mu/Y_x_tyr - Y_tyr_phe*Q_phe

	# these are the rates of change of the amino acids and must be solved simultaneously
	#Q_asp = -mu/Y_x_asp - Y_asp_arg*Q_arg + Y_asp_x
	#Q_arg = -mu/Y_x_arg + Y_arg_glu*Q_glu + Y_arg_pro*Q_pro - Y_arg_asp*Q_asp
	#Q_pro = -mu/Y_x_pro + Y_pro_glu*Q_glu - Y_pro_arg*Q_arg
	#Q_glu = -mu/Y_x_glu + Y_glu_pro*Q_pro - Y_gly_his*Q_his - Y_glu_gln*Q_gln - Y_glu_arg*Q_arg
	# 		 + Y_glu_x 
	#Q_gln = -mu/Y_x_gln - M_gln + Y_gln_glu*Q_glu
	A1 = np.array([[-1, -Y_asp_arg, 0, 0, 0],
				   [-Y_arg_asp, -1, Y_arg_pro, Y_arg_glu, 0],
				   [0, -Y_pro_arg, -1, Y_pro_glu, 0],
				   [0, 0, Y_glu_pro, -1, -Y_glu_gln],
				   [0, 0, 0, Y_gln_glu, -1]   
				  ])
	b1 = np.array([[mu/Y_x_asp - Y_asp_x, 
					mu/Y_x_arg, 
					mu/Y_x_pro, 
					mu/Y_x_glu + Y_glu_his*Q_his -Y_glu_x,
					mu/Y_x_gln + M_gln
				  ]]).T
	x1 = np.linalg.solve(A1, b1)
	Q_asp = x1[0]
	Q_arg = x1[1]
	Q_pro = x1[2]
	Q_glu = x1[3]
	Q_gln = x1[4]

	Q_amm = -Y_amm_gln*Q_gln
	# ODE
	dXv = (mu-mu_d)*Xv
	dXd = (mu_d)*Xd
	dGLN = Q_gln*Xv - K_d_gln*GLN
	dGLC = Q_glc*Xv
	dAMM = Q_amm*Xv + K_d_gln*GLN
	dARG = Q_arg*Xv
	dASP = Q_asp*Xv
	dGLU = Q_glu*Xv
	dHIS = Q_his*Xv
	dPHE = Q_phe*Xv
	dPRO = Q_pro*Xv
	dTYR = Q_tyr*Xv

	return np.array([dXv, dXd, dGLC, dGLN, dAMM, dHIS, dPHE, dTYR, dASP, dARG, dPRO, dGLU])
def objective3(x):
	# decompose the grid search params that the optimize.brute is inputting
	asp, arg, pro, glu, = x
	# make initial condition vector to feed into ODE system to solve for cell density
	ICn = np.append(IC, [asp, arg, pro, glu])
	Xv = integrate.solve_ivp(cells4, (0, t1), ICn, rtol=1e-7) # solve ODEs
	Xv_max = np.max(Xv.y[0]) 	# max cell density
	# find where it is at the max--for what t
	index = np.where(Xv.y[0] == Xv_max)
	t_max = Xv.t[index] 	# t_{max}
	if t_max < 1:
		t_max = 1
	obj = -Xv_max/t_max 	# this is the objective function, return it
	return obj


N = 10
storage = np.empty([N, N, N, N])
print(storage.shape)

asp_grid = np.linspace(hyperparams[0][0], hyperparams[0][1], N)
arg_grid = np.linspace(hyperparams[1][0], hyperparams[1][1], N)
pro_grid = np.linspace(hyperparams[2][0], hyperparams[2][1], N)
glu_grid = np.linspace(hyperparams[3][0], hyperparams[3][1], N)

'''
for i in range(N): 	# iterate thrugh asp
	for j in range(N): 	# iterate through arg
		for k in range(N): # iterate through pro
			for l in range(N): # iterate through glu
				zeta = objective3([asp_grid[i], arg_grid[j], pro_grid[k], glu_grid[l]])
				storage[i][j][k][l] = zeta
				print(i, j, k, l, zeta)

minimum_obj = np.min(storage)
index_min = np.where(storage == minimum_obj)
print(index_min)
print(storage[index_min])
'''
P = optimize.brute(objective3, hyperparams, Ns=4)
print(P)
