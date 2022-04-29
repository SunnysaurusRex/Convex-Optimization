# SZ file_1 (glc, gln)
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

# bounds of each thing in mM
glc = [5, 55]
gln = [0.5, 4]

# make vector of bounds
hyperparams = [glc, gln]

# cell density initial conditions
X0 = 1e6			# inoculum, cell density [=] cells/L
Xv0 = X0*.95		# Assume 95% viability
Xd0 = X0-Xv0 		# assume 5% dead cells or 95% viable cells
t1 = 150			# solve ODEs until 150 hours

# ODE function
def cells1(t, w): # inputs: time value, solution vector at current iteration
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
	# these are independent
	M_gln = a1*GLN/(a2+GLN)

	Q_gln = -mu/Y_x_gln - M_gln
	Q_glc = -mu/Y_x_glc - M_glc
	Q_amm = -Y_amm_gln*Q_gln

	# ODE
	dXv = (mu-mu_d)*Xv
	dXd = (mu_d)*Xd
	dGLN = Q_gln*Xv - K_d_gln*GLN
	dGLC = Q_glc*Xv
	dAMM = Q_amm*Xv + K_d_gln*GLN

	return np.array([dXv, dXd, dGLC, dGLN, dAMM])

def objective(x):
	# decompose the grid search params that the optimize.brute is inputting
	glc, gln = x
	# make initial condition vector to feed into ODE system to solve for cell density
	amm = 1e-10 	# ammonia initial condition, !0 but =~ 0 because it appears in a denom. somewhr
	IC = np.array([Xv0, Xd0, glc, gln, amm])
	Xv = integrate.solve_ivp(cells1, (0, t1), IC, rtol=1e-6) # solve ODEs
	Xv_max = np.max(Xv.y[0]) 	# max cell density
	# find where it is at the max--for what t
	index = np.where(Xv.y[0] == Xv_max)
	t_max = Xv.t[index] 	# t_{max}
	if t_max < 1:
		t_max = 1
	obj = -Xv_max/t_max 	# this is the objective function, return it
	#plt.plot(Xv.t, Xv.y[0])
	return obj

# make 2D grid
N = 20
glc_grid = np.linspace(glc[0], glc[1], N)
gln_grid = np.linspace(gln[0], gln[1], N)

glc_gln = np.empty([N,N])
for i in range(N):
	for j in range(N):
		zeta = objective([glc_grid[i], gln_grid[j]])
		glc_gln[i][j] = zeta		

glc_grid, gln_grid = np.meshgrid(glc_grid, gln_grid, indexing='ij')

plt.figure()
ax = plt.axes(projection='3d')

nx = 4
ny = 3
x = np.arange(0,nx)
y = np.arange(0,ny)

#print(x)
#print(y)

z = np.array([[1, 2, 3, 4],
			 [5, 6, 7, 8],
			 [9, 10, 11,12]
	])
#x, y = np.meshgrid(x, y)
#ax.plot_surface(x, y, z, cmap='viridis')
ax.plot_surface(glc_grid, gln_grid, glc_gln, cmap=cm.coolwarm)
ax.view_init(10, 50)
ax.set_xlabel('glc')
ax.set_ylabel('gln')

minimum_obj = np.min(glc_gln)
index_min = np.where(glc_gln == minimum_obj)
print(index_min)

#print(matrix)
plt.show()

#P = optimize.brute(objective, hyperparams, Ns=5, )
#print(P)

#plt.xlabel('time / hours')
#plt.ylabel('cell density / $cells\\cdot mL^{-1}$')
#plt.title('Viable cell density solution at optimal parameters curve')

#plt.show()