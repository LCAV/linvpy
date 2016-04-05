'''
from linvpy import regression as reg
import generate_random

A,y = generate_random.generate_random_ill_conditioned(3)


print A,y

print reg.tikhonov_regularization(A,y,0)


import numpy as np
from linvpy import regression as reg

A = np.matrix([[7142.80730214, 6050.32000196],
               [6734.4239248, 5703.48709251],
               [4663.22591408, 3949.23319264]])

y = [0.83175086, 0.60012918, 0.89405644]

# Returns x_hat, the tradeoff solution of y = Ax
print reg.tikhonov_regularization(A, y, 50)

# [8.25871731e-05   4.39467106e-05]

import numpy as np
from linvpy import regression as reg

B = np.matrix([[1,3],[3,4],[4,5], [4,5], [4,5]])
t = [-6,1,-2, 2,3]

# Returns x_hat, the least squares solution of y = Ax
print reg.least_squares_gradient(B,t)

import numpy as np
from linvpy import regression as reg

A = np.matrix([[1,3],[3,4],[4,5]])
y = [-6,1,-2]

# Returns x_hat, the least squares solution of y = Ax
print reg.least_squares(A,y)

print np.linalg.lstsq(B,t)[0]
'''

from linvpy import regression as reg

x = [1,2,3,4,5,6,7,8,9]

loss = [reg.phi_huber(e, 4) for e in x]


from linvpy import regression as reg

x = [1,2,3,4,5,6,7,8,9]

output = reg.weight_function(x)

print output






