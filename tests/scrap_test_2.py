import linvpy as lp
import generate_random as gen
import numpy as np
import copy
import random



def gen_noise(rows, columns, coeff_noise=0.5):
    
    matrix_a = np.random.rand(rows,columns)
    vector_x = np.random.rand(columns)
    vector_y = np.dot(matrix_a, vector_x)

    print matrix_a, vector_x, vector_y

    np.random.rand(vector_y.shape[0])
    noise_vector = coeff_noise * np.random.rand(vector_y.shape[0])
    vector_y += noise_vector

    print matrix_a, vector_x, vector_y

    return matrix_a, vector_x, vector_y


gen_noise(3,4,0)

for i in range (0,100,):
	print "iter : ", i
	gen_noise(random.randint(0,i),random.randint(0,i),random.uniform(0.1,3))

'''
noise = vector of shape like y, full of random variables y_noise = 0.5 * np.random.rand(y.shape, 1) + last_y

1) generate A,x
2) generate y = Ax
3) y_noise = y + noise_vector
4) test(y_noise, A)
'''