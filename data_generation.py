import numpy as np
import matplotlib.pyplot as plt

def saveDataToFile(filename, data):
    with open(filename, 'w') as file:
        # Write each pair of numbers to the file
        for pair in data:
            file.write(','.join(map(str, pair)) + '\n')

# Set a random seed for reproducibility
np.random.seed(42)

# Generate correlated data
mean = [0, 0]
covariance_matrix = [[1, 0.8], [0.8, 1]]  # Adjust the covariance to control correlation

# Generate 10,000 data points
data = np.random.multivariate_normal(mean, covariance_matrix, 10000)
saveDataToFile("correlated.txt",data)

# Plot the data
plt.scatter(data[:, 0], data[:, 1], marker='o', s=10)
plt.title('Correlated 2D Dataset (10,000 points)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Generate 10,000 points with a uniform distribution
uniform_data = np.random.uniform(low=-2, high=2, size=(10000, 2))
saveDataToFile("unifrom.txt",uniform_data)

# Plot the data
plt.scatter(uniform_data[:, 0], uniform_data[:, 1], marker='o', s=10)
plt.title('Uniform 2D Dataset (10,000 points)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Generate 10,000 points with a normal distribution
normal_data = np.random.normal(loc=0, scale=1, size=(10000, 2))
saveDataToFile("normal.txt",normal_data)

# Plot the data
plt.scatter(normal_data[:, 0], normal_data[:, 1], marker='o', s=10)
plt.title('Normal 2D Dataset (10,000 points)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Generate anticorrelated data
anticorrelated_covariance_matrix = [[1, -0.8], [-0.8, 1]]

# Generate 10,000 points with an anticorrelated distribution
anticorrelated_data = np.random.multivariate_normal(mean, anticorrelated_covariance_matrix, 10000)
saveDataToFile("anticorrelated.txt",anticorrelated_data)

# Plot the data
plt.scatter(anticorrelated_data[:, 0], anticorrelated_data[:, 1], marker='o', s=10)
plt.title('Anticorrelated 2D Dataset (10,000 points)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

