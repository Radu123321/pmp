
import numpy as np

# Camerele: 1, 2, 3, 4
states = [1, 2, 3, 4]

# Distributia initiala
pi = np.array([0.25, 0.25, 0.25, 0.25])

# Matricea tranzitiilor
A = np.array([
    [0.2, 0.4, 0.0, 0.4],
    [0.4, 0.2, 0.4, 0.0],
    [0.0, 0.4, 0.2, 0.4],
    [0.4, 0.0, 0.4, 0.2]
])

# Matricea emisiilor: red, green, blue
B = np.array([
    [0.7, 0.2, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
    [0.5, 0.2, 0.3]
])

colors = {"red": 0, "green": 1, "blue": 2}

# Exemplu: inlocuieste cu secventa din enunt
observations = ["red", "green", "blue"]
obs = [colors[c] for c in observations]

T = len(obs)
N = len(states)

# delta[t,j] = probabilitatea maxima a unui
# traseu ce se termina in starea j la pasul t
delta = np.zeros((T, N))

# psi[t,j] = starea precedenta optima
psi = np.zeros((T, N), dtype=int)

# Initializare
delta[0] = pi * B[:, obs[0]]

# Recurenta Viterbi
for t in range(1, T):
    for j in range(N):
        valori = delta[t - 1] * A[:, j]

        psi[t, j] = np.argmax(valori)
        delta[t, j] = np.max(valori) * B[j, obs[t]]

# Probabilitatea comuna maxima
probabilitate = np.max(delta[T - 1])

# Ultima stare optima
last_state = np.argmax(delta[T - 1])

# Reconstruim secventa
path = [last_state]

for t in range(T - 1, 0, -1):
    last_state = psi[t, last_state]
    path.append(last_state)

path.reverse()

# Convertim indicii 0..3 in camerele 1..4
camere = [states[i] for i in path]

print("Secventa cea mai probabila:", camere)
print("Probabilitatea comuna:", probabilitate)
print("Matricea delta:")
print(delta)
