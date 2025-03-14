import numpy as np
from timeit import default_timer as timer
from exTime import cost_function, gradient

"""before optimization"""
# Number of times to run predict function
num_runs = 10
cost_times = []
grad_times = []

input_layer_size = 5000
hidden_layer_size = 8
num_labels = 3
m = 100  # Number of training samples
lmbda = 1.0      # Example input data (100 samples, 5000 features)

X = np.random.rand(m, input_layer_size)
y = np.random.randint(0, num_labels, size=(m, 1))
theta = np.random.rand((hidden_layer_size * (input_layer_size + 1)) + (num_labels * (hidden_layer_size + 1)))

# Measure execution time for cost_function
for _ in range(num_runs):
    start_time = timer()
    cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    end_time = timer()
    cost_times.append(end_time - start_time)

# Measure execution time for gradient
for _ in range(num_runs):
    start_time = timer()
    gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    end_time = timer()
    grad_times.append(end_time - start_time)

# Compute average and standard deviation
avg_cost_time = np.mean(cost_times)
std_cost_time = np.std(cost_times)
avg_grad_time = np.mean(grad_times)
std_grad_time = np.std(grad_times)

# Print results
print(f"Cost Function - Average Execution Time: {avg_cost_time:.6f} seconds, Standard Deviation: {std_cost_time:.6f}")
print(f"Gradient Function - Average Execution Time: {avg_grad_time:.6f} second, Standard Deviation: {std_grad_time:.6f}")
