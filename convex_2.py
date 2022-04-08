# SZ 
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def f(t):
	x, y = t
	return x**2 + y**2

interval = ((-10, 3), (1, 4))

inte = []
int1 = [-10, 3]
int2 = [1, 4]

inte.append(int1)
print(inte)
inte.append(int2)
print(inte)
soln = optimize.brute(f, inte, Ns=30)

print(soln)