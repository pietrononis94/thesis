import numpy as np
from scipy.optimize import linprog

# Set the inequality constraints matrix
# Note: the inequality constraints must be in the form of <=
A = np.array([[1, 0], [2, 3], [1, 1], [-1, 0], [0, -1]])

# Set the inequality constraints vector
b = np.array([16, 19, 8, 0, 0])

# Set the coefficients of the linear objective function vector
# Note: when maximizing, change the signs of the c vector coefficient
c = np.array([-5, -7])

# Solve linear programming problem
res = linprog(c, A_ub=A, b_ub=b)

# Print results
print('Optimal value:', round(res.fun*-1, ndigits=2),
      '\nx values:', res.x,
      '\nNumber of iterations performed:', res.nit,
      '\nStatus:', res.message)
