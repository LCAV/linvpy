import linvpy as lp
import mestimator_marta as marta
import generate_random as gen
import numpy as np
import matplotlib.pyplot as plt
import optimal as opt
from scipy.sparse.linalg import lsmr
import toolboxutilities as util
import toolboxinverse as inv
import copy



#genA, geny = gen.generate_random(4,5)
#print "test =", lp.irls(genA, geny, lp.psi_huber)

#28.04.16

# I fix here the number of measurements of vector y
nmeasurements = 5

# I define a vector x to use it in my functions to generate the matrix A and vector y
x_base = np.ones((2, 1))  # fixed source

# I generate the matrix A
a_base = util.getmatrix(2, 'random', nmeasurements)  # get the sensing matrix

b_base = np.matrix([[0.18556994, 0.85051253],
	[-0.10116954, 1.27607103],
 [-1.21054471, 0.78025321],
 [ 0.41563372, 1.01907658],
 [ 0.77237054, -0.33484395]])

# I generate the vector y
y_base = util.getmeasurements(a_base, x_base, 'gaussian')

'''
y_base = np.matrix([[-3.30284787],
 [-0.82014103],
 [ 1.26663908],
 [-0.04444286],
 [-0.82479587]])

a_base = np.matrix([[-1.29147407, -1.96845515],
 [-0.55629565, 0.09258146],
 [ 1.09840683, 0.73997283],
 [ 0.06732681, -0.6932167 ],
 [ 1.83309306, -1.08687388]])
'''

y_gui = copy.deepcopy(y_base)
a_gui = copy.deepcopy(a_base)


y_marta = copy.deepcopy(y_base)
a_marta = copy.deepcopy(a_base)


print "MARTA'S INPUTS : "
print y_marta
print a_marta


#y_gui = np.copy(y_marta)
#a_gui = np.copy(a_marta)


print "GUILLAUME'S INPUTS : "
print y_gui
print a_gui



# define parameters necessary for basic tau...
lossfunction = 'optimal'

# we need two because in the tau estimator we build the rho functin wiht other two
clipping_parameters = (0.4, 1.09)

# how many initial solutions do we want to try
n_initial_solutions = 10

# max number of iterations for irls
max_iter = 10

# how many solutions do we keep
n_best = 3

# called the basic tau estimator
xhat, shat = inv.basictau(
  y_marta,
  a_marta,
  lossfunction,
  clipping_parameters,
  n_initial_solutions,
  max_iter,
  n_best
)




# check what we got back. we should get n_best xhats
print "MARTA's tau : "
print xhat
print shat

print "Guillaume's y,a -1 = ", y_gui, a_gui 

xhat2, shat2 = lp.basictau(
  y_gui,
  a_gui,
  lossfunction,
  clipping_parameters,
  n_initial_solutions,
  max_iter,
  n_best
)



print "GUILLAUME's tau : "
print xhat2
print shat2
print (xhat.all() == xhat2.all()) and (shat2.all() == shat.all())



'''

c = np.ones((2, 1))
d = np.ones((2, 2))

print "C = ", c
print "D = ", d


y_gui = x = np.array([3,5])
A_gui = np.array([[1, 2], [3, 4]])



new_y = np.array([[2],[3]])
new_a = np.array([[3,3],[2,3]])
m, n = new_a.shape
initialx = np.ones((n, 1))  # initial solution

print "Marta's M-estimator = ", marta.mestimator(new_y, new_a, 'huber',0.5)[0]

print "MARTA's IRLS = ", marta.irls(new_y, new_a, 'huber', initialx, 0.5)[0]


last_y = [2,3]
last_a = np.array([[3,3],[2,3]])

new_y = [1,1]
new_a = np.array([[1,1],[1,1]])

# print "My m-estimator = ", lp.irls(last_a, last_y, lp.psi_huber, clipping=1.5,
#	lamb=0, scale=100, initial_x=[1,4])


print "My m-estimator = ", lp.irls(last_a, last_y, lp.rho_optimal, clipping=1.5,
	lamb=0, scale=4, initial_x=[-5,4], kind=None)



# def mestimator(y, a, lossfunction, clipping):

# for a large clipping parameter, LS and M outputs should be the same


'''







'''
print "my psi_huber =", [reg.psi_huber(e,3) for e in x]
print "x=", x
print "my weights =", reg.weights(x, reg.psi_huber, 3)

print marta.weights(B, 'M', 'huber', 1.5, 0)

print reg.rho_bisquare(2,5)
print [reg.rho_bisquare(i,5) for i in x]


plt.plot([reg.psi_cauchy(i) for i in range(-20,20)])
plt.show()

print [reg.psi_bisquare(i,5) for i in x]

marta_list = opt.rhooptimal(y,2)
my_list = [lp.rho_optimal(i) for i in y]

marta_numpy = np.array(marta_list)
my_numpy = np.array(my_list)

print "marta's rho : ", opt.rhooptimal(y,2)
print "marta's score : ", opt.scoreoptimal(y,2)
#print reg.rho_optimal(2,3)

print "mine : ", [lp.rho_optimal(i) for i in y]


A,y = gen.generate_random(4)
#print "my gradient= ", lp.least_squares_gradient(A,y)
#print "my matrice form= ", lp.least_squares(A,y) 
#print "numpy's =", np.linalg.lstsq(A,y)[0]

LAMBDA=0.2
print lp.tikhonov_regularization(A,y,LAMBDA)
print lsmr(A,y,LAMBDA)[0]


x = [1,2,3,4,5,6,7,8,9]

result = [lp.psi_cauchy(e, 4) for e in x]

# [0.9411764705882353, 1.6, 1.92, 2.0, 1.951219512195122, 1.8461538461538463, 1.7230769230769232, 1.6, 1.4845360824742269]

print result

def lol(a,b):
	return a+b

def func(function,a,b):
	return function(a,b)


'''

'''
A,y = gen.generate_random(5,4)



# TEST WEIGHTS => SAME BUT DIVIDED BY 2

print "my weights =", lp.weights(y, lp.psi_huber)

print "marta's weights = ", marta.weights(y, 'M', 'huber', 0.1, 0)
'''

