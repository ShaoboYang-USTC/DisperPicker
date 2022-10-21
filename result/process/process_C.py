import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../../config')
from config import Config
# from config.config import Config

# read file
root_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(root_path)[0] + '/pick_result/phase_velocity'
os.system('rm -rf %s/new/phase_velocity/*'%root_path)

# station loaction
station = {}
with open('../../config/station.txt', 'r') as f:
    for each_line in f.readlines():
        item = each_line.split()
        station[item[0]] = [float(item[1]), float(item[2])]

# process 1: dist must be larger than 1.5*v*T
range_T = Config().range_T
T = np.linspace(range_T[0], range_T[1], range_T[2])
os.chdir(path)
dir_name = []
velocity = []
all_dir = glob.glob('*txt')
all_dir.sort()
all_dist = []
truncated = 0
for file in all_dir:
    sta_pair = file.split('.')[:2]
    loc1 = station[sta_pair[0]]
    loc2 = station[sta_pair[1]]
    dist = (111*((loc1[0] - loc2[0])**2 + ((loc1[1] - loc2[1])*math.cos(loc1[0]*
                3.14/180))**2)**0.5)
    all_dist.append(dist)
    each_velocity = np.loadtxt(file)[:, 1]
    for i in range(len(T)):
        if each_velocity[i] != 0:
            # print(dist / 1.5 / each_velocity[i], T[i])
            if dist/1.5/each_velocity[i] < T[i]:
                truncated += 1
                break
    new_each_v = np.zeros(len(T))
    new_each_v[:i+1] = each_velocity[:i+1]
    
    dir_name.append(file)
    velocity.append(new_each_v)
print('Truncated number:', truncated)
# print(min(all_dist))

# process
# process 2: remove outliers
v_max = 3.8           # Should be modified
v_min = 0.6           # Should be modified
num = 0
all_group_velocity_num = 0
for i in range(len(velocity)):
    for j in range(len(velocity[i])):
        if velocity[i][j] != 0:
            if velocity[i][j] > v_max or velocity[i][j] < v_min:
                num = num+1
                # velocity[i] = np.zeros(range_T[2])
                # break
                velocity[i][j] = 0

# print(num)

# process 3: find the most reliable stage.
downward = -0.1                 # The velocity of the curve must decreases less than 0.3 km/s from short to long period. 
each_stage_downward = 0.2      # Each of the valid stage of the curve must decreases less than 0.2 km/s.
max_diff = 0.1                 # v[i+1] - v[i] <= max_diff
min_diff = -0.08                # v[i+1] - v[i] >= min_diff
min_start_t = 20    # npts not (s). Delete the curve if it start after the 20 th point.
min_len = 15        # Delete the curve if the number of points of the curve is less than 15.
    
for i in range(len(velocity)):
    #print('======================================')
    stage = np.zeros(len(velocity[i]))
    for j in range(len(velocity[i]) - 1):
        if velocity[i][j] != 0 and velocity[i][j+1] != 0:
            if (velocity[i][j+1] - velocity[i][j] < max_diff and 
                    velocity[i][j+1] - velocity[i][j] > min_diff):
                stage[j] = 1
            else:
                stage[j] = -1
        elif velocity[i][j] != 0 and velocity[i][j+1] == 0:
            stage[j] = 1

    # print(dir_name[i])
    # print(stage)

    if (stage[-2] == 1 and velocity[i][-1] - velocity[i][-2] < max_diff and
            velocity[i][-1] - velocity[i][-2] > min_diff):
        stage[-1] = 1

    for j in range(len(stage))[2:-2]:
        if (stage[j] == -1 and stage[j-1] == 1 and stage[j-2] == 1 and 
                stage[j+1] == 1 and stage[j+2] == 1):
            stage[j] = 1

    start = []
    end = []
    new_stage = True
    for j in range(len(stage)):
        if stage[j] == 1 and new_stage == True:
            start.append(j)
            new_stage = False
        if stage[j] != 1 and new_stage == False:
            end.append(j-1)
            new_stage = True
        if stage[j] == 1 and new_stage == False and j == len(stage)-1:
            end.append(j)
    stage_len = []
    start0 = []
    end0 = []
    for j in range(len(start)):
        if start[j] <= min_start_t:
            stage_len.append(end[j] - start[j] + 1)
            start0.append(start[j])
            end0.append(end[j])
    #print(start,end,stage_len)
    #print('name: ',dir_name[i])
    #print('velocity: \n',velocity[i])
    new_velocity = np.zeros(len(velocity[i]))

    # sort to use larger stage prefer
    if len(stage_len) > 1:
        info = [stage_len,start0,end0]
        #print(info)
        info = np.array(info)
        info = info.T
        info = info.tolist()
        info.sort()
        info.reverse()
        info = np.array(info)
        info = info.T
        info = info.tolist()
        #print(info)
        stage_len = info[0]
        start = info[1]
        end = info[2]

    for k in range(len(stage_len)):
        #print(velocity[i][start[max_stage]],velocity[i][end[max_stage]])
        if stage_len[k] >= min_len:
            valid_velocity = velocity[i][start[k]:end[k]+1]
            #print(valid_velocity)
            start2 = [0]
            end2 = []
            for j in range(len(valid_velocity))[1:-1]:
                if valid_velocity[j] >= valid_velocity[j-1] and valid_velocity[j] >= valid_velocity[j+1]:
                    end2.append(j)
                    start2.append(j)
                elif valid_velocity[j] <= valid_velocity[j-1] and valid_velocity[j] <= valid_velocity[j+1]:
                    end2.append(j)
                    start2.append(j)
            end2.append(len(valid_velocity) - 1)
            valid = []
            for j in range(len(start2)):
                if valid_velocity[start2[j]] - valid_velocity[end2[j]] < each_stage_downward:
                    valid.append(1)
                else:
                    valid.append(0)
            #print('stages:',start2,end2,valid)
            new_stage = True
            valid_start2 = []
            valid_end2 = []
            for j in range(len(valid)):
                if valid[j] == 1 and new_stage == True:
                    valid_start2.append(j)
                    new_stage = False
                if valid[j] != 1 and new_stage == False:
                    valid_end2.append(j - 1)
                    new_stage = True
                if valid[j] == 1 and new_stage == False and j == len(valid) - 1:
                    valid_end2.append(j)
            #print('valid:',valid_start2,valid_end2)
            true_start = []
            true_end = []
            true_length = []
            for j in range(len(valid_start2)):
                true_start.append(start2[valid_start2[j]])
                true_end.append(end2[valid_end2[j]])
                true_length.append(end2[valid_end2[j]] - start2[valid_start2[j]] + 1)
            #print('true:',true_start,true_end,true_length)
            if len(true_length) > 0:
                max_index = true_length.index(max(true_length))
                final_velocity = np.zeros(len(valid_velocity))
                final_velocity[true_start[max_index]:
                                   true_end[max_index]+1] = valid_velocity[true_start[max_index]:true_end[max_index]+1]
                #print(final_velocity)
                if (true_length[max_index] >= min_len and 
                        final_velocity[true_start[max_index]] - final_velocity[true_end[max_index]] < downward):
                    none_zero = np.where(final_velocity != 0)[0]
                    if min(none_zero) <= min_start_t:
                        all_group_velocity_num += 1
                        new_velocity[start[k]:end[k]+1] = final_velocity
                    #print(new_velocity)

                    write_G = []
                    write_G.append(T)
                    write_G.append(new_velocity)
                    write_G = np.array(write_G).T
                    np.savetxt(root_path + '/new/phase_velocity/' + dir_name[i], write_G, fmt="%1.2f  %1.2f")
                    break

    velocity[i] = new_velocity

print('remaining phase velocity number: ', all_group_velocity_num)

# plot
fontsize = 20
figformat = '.png'
plt.figure(figsize=(8, 5), clear=True)

all_color = ['-k','-b','-g','-y','-r','-m','-c']
for k in range(len(velocity))[:300]:
    start = 0
    end = len(velocity[k])
    for i in range(len(velocity[k])):
        if velocity[k][i] != 0:
            start = i
            break
    for j in range(len(velocity[k]))[start:]:
        if velocity[k][j] == 0:
            end = j
            break
    color = random.sample(all_color, 1)[0]
    plt.plot(T[start:end], velocity[k][start:end], color, linewidth=1.5)
    #if k == 15:
    #    break

plt.xlabel('Period (s)', fontsize=fontsize)
plt.ylabel('Phase Velocity (km/s)', fontsize=fontsize)
plt.ylim((0, 5))
plt.yticks([0, 1, 2, 3, 4, 5])
plt.title('Phase velocity', fontsize=fontsize)
plt.tick_params(labelsize=15)

plt.savefig(root_path + '/DP_C.jpg', bbox_inches='tight', dpi=600)
plt.close()













