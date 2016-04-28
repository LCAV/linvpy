'''
    Generation of some random matrix, vectors etc for test purpose.
'''

import numpy as np

CONDITION_NUMBER_LOWERBOUND = 10000

def generate_random(rows,columns):
    '''
    :param size: (int) size of matrix and vector

    :return tuple(np.matrix, array): a random tuple A,y of matching dimensions
    '''

    if rows == 1:
        return np.array([[np.random.rand()]]), [np.random.rand()]
    return np.array(np.random.rand(rows,columns)) , np.random.rand(rows)


def generate_random_ill_conditioned(size):
    '''
    For test purpose only. Function generating a random ill-conditioned matrix
    of the size given in parameter.

    :param size: (int) size of matrix and vector

    :return tuple(np.matrix, array): a random tuple A,y of matching 
    dimensions with A being an ill-conditioned matrix
    '''

    # An ill-conditioned matrix of size 1 makes no sense so it calls the
    # normal random generator
    if(size == 1):
        return generate_random(1)

    # Generates a random <size,size-1> matrix
    random_matrix = np.matrix(np.random.rand(size,size-1))

    # Unitary_1, unitary_2 are np.matrix types ; singular is a vector
    unitary_1, singular, unitary_2 = np.linalg.svd(random_matrix, 
        full_matrices=False)

    # Finds the largest singular value, multiplies it by
    # CONDITION_NUMBER_LOWERBOUND and put it as the first element of the
    # singular values vector to make sure the condition number is greater
    # than CONDITION_NUMBER_LOWERBOUND
    max_value = max(singular)
    singular[0]= max_value * CONDITION_NUMBER_LOWERBOUND

    # Diagnolize the singular vector
    S = np.diag(singular)

    # Recomposes the random matrix which is ill-conditioned now
    ill_conditioned_matrix = np.dot(
        unitary_1, 
        np.dot(S, unitary_2)
        )

    # Generates a random vector <size>
    random_vector = np.random.rand(size)

    return ill_conditioned_matrix, random_vector
