import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../../config')
from config import Config

config = Config()
root = config.root
sub = '/result/process/new'
all_file = os.listdir(root + sub + '/group_velocity')
for file in os.listdir(root + sub + '/phase_velocity'):
    if file not in all_file:
        all_file.append(file)
if config.test:
    for file in os.listdir(root + '/data/TestData/group_velocity'):
        if file not in all_file:
            all_file.append(file)
    for file in os.listdir(root + '/data/TestData/phase_velocity'):
        if file not in all_file:
            all_file.append(file)
print(len(all_file))
# part_file = random.sample(all_file,200)
part_file = all_file
part_file.sort()

range_T = config.range_T
range_V = config.range_V
col = range_T[-1]
row = range_V[-1]
i=0
for file in part_file:
    file = os.path.splitext(file)[0]
    key = file.split('.')
    i += 1
    print(i)
    group_image = np.loadtxt(root + '/data/TestData/group_image/' + file + '.dat')[:, :col]
    group_image2 = np.zeros((row, col))
    length = len(group_image)
    group_image2[row-length:] = group_image
    phase_image = np.loadtxt(root + '/data/TestData/phase_image/' + file + '.dat')[:, :col]
    phase_image2 = np.zeros((row, col))
    length = len(phase_image)
    phase_image2[row-length:] = phase_image
    
    T = np.linspace(range_T[0], range_T[1], range_T[2])
    V = np.linspace(range_V[0], range_V[1], range_V[2])

    plt.figure(num=1, figsize=(10, 2.5), dpi=300, clear=True)
    fontsize = 12
    figformat = '.png'
    plt.tick_params(labelsize=5)

    plt.subplot(121)
    z_max = group_image2.max()
    z_min = group_image2.min()
    plt.pcolor(T, V, group_image2, cmap='jet', vmin=z_min, vmax=z_max + 0.05)
    plt.colorbar()

    if config.test and file + '.txt' in os.listdir(root + '/data/TestData/group_velocity'):
        start = 0
        end = col - 1
        group_velocity = np.loadtxt(root + '/data/TestData/group_velocity/' + file + '.txt')[:, 1]
        for k in range(len(group_velocity)):
            if group_velocity[int(k)] != 0:
                start = int(k)
                break
        for j in range(len(group_velocity))[start:]:
            if group_velocity[int(j)] == 0:
                end = int(j)
                break
        plt.plot(T[start:end], group_velocity[start:end], '-w', linewidth=2, label='Manually Picked')
        start = 0
        end = col - 1

    if file + '.txt' in os.listdir(root + sub + '/group_velocity'):
        start = 0
        end = col - 1
        group_velocity = np.loadtxt(root + sub + '/group_velocity/' + file + '.txt')[:, 1]
        for k in range(len(group_velocity)):
            if group_velocity[int(k)] != 0:
                start = int(k)
                break
        for j in range(len(group_velocity))[start:]:
            if group_velocity[int(j)] == 0:
                end = int(j)
                break
        plt.plot(T[start:end], group_velocity[start:end], '--k', linewidth=2, label='DisperPicker')
        start = 0
        end = col - 1

    plt.legend(loc=0)
    plt.xlabel('Period (s)', fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)', fontsize=fontsize)
    plt.title(key[0] + '-' + key[1], fontsize=fontsize + 1)

    plt.subplot(122)
    z_max = phase_image2.max()
    z_min = phase_image2.min()
    plt.pcolor(T, V, phase_image2, cmap='jet', vmin=z_min, vmax=z_max + 0.05)
    plt.colorbar()

    if config.test and file + '.txt' in os.listdir(root + '/data/TestData/phase_velocity'):
        start = 0
        end = col - 1
        phase_velocity = np.loadtxt(root + '/data/TestData/phase_velocity/' + file + '.txt')[:, 1]
        for k in range(len(phase_velocity)):
            if phase_velocity[int(k)] != 0:
                start = int(k)
                break
        for j in range(len(phase_velocity))[start:]:
            if phase_velocity[int(j)] == 0:
                end = int(j)
                break
        plt.plot(T[start:end], phase_velocity[start:end], '-w', linewidth=2, label='Manually Picked')
        start = 0
        end = col - 1

    if file + '.txt' in os.listdir(root + sub + '/phase_velocity'):
        phase_velocity = np.loadtxt(root + sub + '/phase_velocity/' + file + '.txt')[:, 1]
        start = 0
        end = col - 1
        for k in range(len(phase_velocity)):
            if phase_velocity[int(k)] != 0:
                start = int(k)
                break
        for j in range(len(phase_velocity))[start:]:
            if phase_velocity[int(j)] == 0:
                end = int(j)
                break
        plt.plot(T[start:end], phase_velocity[start:end], '--k', linewidth=2, label='DisperPicker')
        start = 0
        end = col - 1

    plt.legend(loc=0)
    plt.xlabel('Period (s)', fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)', fontsize=fontsize)
    plt.title(key[0] + '-' + key[1], fontsize=fontsize + 1)
    name = file.split('.')
    plt.savefig(root + sub + '/plot/' + name[0] + '.' + name[1] + '.jpg', bbox_inches='tight', dpi=300)
    plt.close()
















