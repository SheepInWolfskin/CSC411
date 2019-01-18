import matplotlib.pyplot as plt
import numpy as np




# Huber Loss function
x = np.linspace(-10,10,100)
# define delta by yourself 4 is just an example
delta = 6
y = 0.5 * (x**2) * (np.abs(x) <= delta) + np.multiply(delta, np.abs(x) - 0.5 * delta) * (np.abs(x) > delta)
plt.plot(x, y, '-')
plt.show()


# LSE
x = np.linspace(-10, 10, 100)
LSE = 0.5 * (x ** 2)
plt.plot(x, LSE, '-')
plt.show()


