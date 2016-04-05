#import profile
#import cProfile
#import re
#from linear_inverse import regression
#import generate_random as gr 

# cProfile.run('re.compile("foo|bar")')
cProfile.run('gr.generate_random(10)')

A,y = gr.generate_random(10)
cProfile.run('regression.least_squares(A,y)')
cProfile.run('regression.least_squares_gradient(A,y)')