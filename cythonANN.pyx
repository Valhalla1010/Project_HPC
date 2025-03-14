import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp

np.import_array()

# Activation function
cpdef np.ndarray[np.float64_t, ndim=2] sigmoid(np.ndarray[np.float64_t, ndim=2] values):
    return 1.0 / (1.0 + np.exp(-values))

# Derivative of activation function
cpdef np.ndarray[np.float64_t, ndim=2] sigmoid_derivative(np.ndarray[np.float64_t, ndim=2] values):
    cdef np.ndarray[np.float64_t, ndim=2] sig_val = sigmoid(values)
    return sig_val * (1.0 - sig_val)

# Reshape theta into weight matrices
cdef tuple reshape_weights(np.ndarray[np.float64_t, ndim=1] weights, unsigned int input_size, unsigned int hidden_size, unsigned int output_size):
    cdef unsigned int split_point = hidden_size * (input_size + 1)
    cdef np.ndarray[np.float64_t, ndim=2] W1 = weights[:split_point].reshape((hidden_size, input_size + 1), order='C')
    cdef np.ndarray[np.float64_t, ndim=2] W2 = weights[split_point:].reshape((output_size, hidden_size + 1), order='C')
    return W1, W2

@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] compute_gradient(np.ndarray[np.float64_t, ndim=1] weights,
                                                        unsigned int input_size,
                                                        unsigned int hidden_size,
                                                        unsigned int output_size,
                                                        np.ndarray[np.float64_t, ndim=2] features,
                                                        np.ndarray labels,
                                                        double reg_param):
    cdef np.ndarray[np.float64_t, ndim=2] W1, W2
    W1, W2 = reshape_weights(weights, input_size, hidden_size, output_size)

    cdef unsigned int sample_count = labels.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] grad_W1 = np.zeros((hidden_size, input_size + 1))
    cdef np.ndarray[np.float64_t, ndim=2] grad_W2 = np.zeros((output_size, hidden_size + 1))

    cdef np.ndarray[np.float64_t, ndim=2] input_layer = np.vstack((np.ones((1, features.shape[0])), features.T))
    cdef np.ndarray[np.float64_t, ndim=2] hidden_layer_input = np.dot(W1, input_layer)
    cdef np.ndarray[np.float64_t, ndim=2] hidden_layer_output = sigmoid(hidden_layer_input)
    hidden_layer_output = np.vstack((np.ones((1, features.shape[0])), hidden_layer_output))
    cdef np.ndarray[np.float64_t, ndim=2] output_layer = sigmoid(np.dot(W2, hidden_layer_output))  

    cdef np.ndarray[np.float64_t, ndim=2] expected_output = np.zeros((output_size, features.shape[0]))
    for idx in range(sample_count):
        expected_output[labels[idx, 0].astype(int), idx] = 1.0

    cdef np.ndarray[np.float64_t, ndim=2] error_output = output_layer - expected_output 
    grad_W2 += np.dot(error_output, hidden_layer_output.T) 

    cdef np.ndarray[np.float64_t, ndim=2] error_hidden = np.dot(W2[:, 1:].T, error_output) * sigmoid_derivative(hidden_layer_input) 
    grad_W1 += np.dot(error_hidden, input_layer.T)   

    cdef np.ndarray[np.float64_t, ndim=2] W1_adj = grad_W1 / sample_count
    cdef np.ndarray[np.float64_t, ndim=2] W2_adj = grad_W2 / sample_count

    W1_adj[:, 1:] += (reg_param / sample_count) * W1[:, 1:]  
    W2_adj[:, 1:] += (reg_param / sample_count) * W2[:, 1:]

    cdef np.ndarray[np.float64_t, ndim=1] final_gradient = np.concatenate((W1_adj.flatten(), W2_adj.flatten()))

    return final_gradient
