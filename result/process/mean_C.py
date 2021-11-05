import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../config')
from config import Config

range_T = Config().range_T
root_path = os.path.abspath(os.path.dirname(__file__))
dir = root_path + '/new/phase_velocity'
os.chdir(dir)
velocity = np.zeros(range_T[2])
max_v = np.zeros(range_T[2])
min_v = np.zeros(range_T[2])
num = np.zeros(range_T[2])
mean = np.zeros(range_T[2])

for file in glob.glob('*txt'):
    v = np.loadtxt(file)[:, 1]
    i = 0
    for each in v:
        if each != 0:
            velocity[i] = velocity[i] + each
            num[i] += 1
        i += 1

for i in range(len(velocity)):
    if num[i] != 0:
        mean[i] = velocity[i]/num[i]
print(num)
print(mean)
fontsize = 18
figformat = '.png'

T = np.linspace(range_T[0], range_T[1], range_T[2])
plt.plot(T, mean, '-go', linewidth=2, markersize=3)

plt.xlabel('T(s)', fontsize=fontsize)
plt.ylabel('v(km/s)', fontsize=fontsize)
plt.ylim((0, 4))
plt.title('Phase velocity', fontsize=fontsize)
plt.tick_params(labelsize=15)

plt.savefig(root_path + '/mean_C', bbox_inches='tight', dpi=300)
plt.close()



