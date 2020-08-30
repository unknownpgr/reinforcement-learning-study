from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import numpy as np

W = 50
H = 50


def leakyReLU(x):
    x[x < 0] *= 0.1
    return x


def dleakyReLU(x):
    x[x > 0] = 1
    x[x < 0] = 0.1
    return x


class NN:
    def __init__(self, layerStructure):
        self.weights = []
        for i in range(len(layerStructure)-1):
            indim = layerStructure[i]+1
            outdim = layerStructure[i+1]+1
            layer = np.random.rand(indim, outdim)*2/outdim-1
            self.weights.append(layer)

    def calc(self, x):
        t = x+[1]
        for layer in self.weights[:-1]:
            t = np.matmul(t, layer)
            t = leakyReLU(t)
        t = np.matmul(t, self.weights[-1])
        return t

    def backprop(self, x, err, alpha=0.1):
        '''
        dE/dW = f-1
        그러므로 
        '''


class AQ_Function:
    def __init__(self):
        self.table = np.zeros([H, W])

    def calc(self, state):
        x = state[0]
        y = state[1]
        table = self.table
        # NESW
        return [
            table[(y-1) % H, x],
            table[y, (x+1) % W],
            table[(y+1) % H, x],
            table[y, (x-1) % W]
        ]

    def update(self, befState, newValue):
        self.table[befState[1], befState[0]] = newValue


def selectOne(values):
    M = -99999
    r = []
    for i in range(len(values)):
        v = values[i]
        if v > M:
            M = v
            r = [i]
        if v == M:
            r.append(i)
    if len(r) == 1:
        return r[0]
    return r[random.randint(0, len(r)-1)]


class AQ_Agent:
    def __init__(self, function):
        self.function = function
        self.state = [0, 0]

    def move(self, dest):
        action = selectOne(self.function.calc(self.state))
        befState = self.state
        if action == 0:
            self.state[1] -= 1
        elif action == 1:
            self.state[0] += 1
        elif action == 2:
            self.state[1] -= 1
        elif action == 3:
            self.state[0] -= 1
        self.state[0] %= W
        self.state[1] %= H
        newValue = np.max(self.function.calc(self.state))-1
        if self.state[0] == dest[0] and self.state[1] == dest[1]:
            self.function.update(befState, -1)
            return True
        self.function.update(befState, newValue)
        return False


func = AQ_Function()
agent = AQ_Agent(func)

print(agent.state)
for i in range(50000):
    if agent.move([W-3, H-3]):
        agent.state = [random.randint(0, W-1), random.randint(0, H-1)]


fmap = agent.function.table

with open('r.csv', 'w') as o:
    for row in fmap:
        for value in row:
            o.write(f'{value},')
        o.write('\n')


X = np.arange(0, W, 1)
Y = np.arange(0, H, 1)
X, Y = np.meshgrid(X, Y)
Z = fmap

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
plt.show()
