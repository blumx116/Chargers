import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from env.region_bounds import gjc02_region_bounds as b
from matplotlib import cm
import numpy as np
fig, ax = plt.subplots(1)

cmap = plt.get_cmap('Pastel1')
for i, key in enumerate(b.keys()):
    ps = b[key]
    xy = (ps['left'], ps['bottom'])
    width = ps['right'] - ps['left']
    height = ps['top'] - ps['bottom']
    print(f"{xy} : {width} x {height}")
    color = cmap.colors[i]
    rect = Rectangle(xy, width, height, linewidth=1, label=key, edgecolor=color, facecolor=color, fill=True)
    ax.add_patch(rect)

ax.set_xlim(left=100, right=125)
ax.set_ylim(20, 47)
ax.legend(b.keys())
plt.show()