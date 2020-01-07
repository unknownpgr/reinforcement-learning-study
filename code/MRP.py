import numpy as np

transition = np.array([
    [0,   0.5, 0,   0,   0,   0.5, 0],
    [0,   0,   0.8, 0,   0,   0,   0.2],
    [0,   0,   0,   0.6, 0.4, 0,   0],
    [0,   0,   0,   0,   0,   0,   1],
    [0.2, 0.4, 0.4, 0,   0,   0,   0],
    [0.1, 0,   0,   0,   0,   0.9, 0],
    [0,   0,   0,   0,   0,   0,   1]
])

r = np.array([-2, -2, -2, 10, 1, -1, 0])
k = np.zeros([7, 7])
transitionPower = np.identity(7)
gamma = 1

for i in range(7):
    s = np.array([0, 0, 0, 0, 0, 0, 0])
    s[i] = 1

    for i in range(1000):
        k += transitionPower*(gamma**(i))
        transitionPower = np.matmul(transitionPower, transition)

    print(round(np.dot(np.matmul(s, k), r), 1))
