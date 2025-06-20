import numpy as np
import matplotlib.pyplot as plt

def plotWealth(data):
    wealths = [[] for _ in data[0]]
    for round in data:
        for i, player in enumerate(round):
            wealth = np.sum(player[1])
            wealths[i].append(wealth)

    fig, ax = plt.subplots()

    for player in wealths:
        ax.plot(player)

    plt.show(block=True)