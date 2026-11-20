
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior initial
a, b = 2, 8

# Primul lot
n1, k1 = 20, 5

# Al doilea lot
n2, k2 = 15, 3

# a) Posterior dupa primul lot
a1 = a + k1
b1 = b + n1 - k1

# b) Posterior dupa ambele loturi
a12 = a + k1 + k2
b12 = b + (n1 - k1) + (n2 - k2)

# c) Posterior folosind doar al doilea lot
a2 = a + k2
b2 = b + n2 - k2

print("a) Beta:", a1, b1)
print("Media:", a1 / (a1 + b1))

print("b) Beta:", a12, b12)
print("Media:", a12 / (a12 + b12))

print("c) Beta:", a2, b2)
print("Media:", a2 / (a2 + b2))

# Reprezentare grafica
x = np.linspace(0, 1, 1000)

plt.plot(x, beta.pdf(x, a, b),
         label="Prior Beta(2,8)")

plt.plot(x, beta.pdf(x, a1, b1),
         label="Lot 1: Beta(7,23)")

plt.plot(x, beta.pdf(x, a12, b12),
         label="Ambele loturi: Beta(10,35)")

plt.plot(x, beta.pdf(x, a2, b2),
         label="Doar lotul 2: Beta(5,20)")

plt.xlabel("Probabilitatea de defect")
plt.ylabel("Densitate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
