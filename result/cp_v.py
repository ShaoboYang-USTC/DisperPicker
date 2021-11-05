import os
import glob
import numpy as np

v_max = 5
v_min = 0.5
points = 15
i = 0
root = os.path.abspath(os.path.dirname(__file__))
os.system('rm -rf %s/raw/group_velocity/*'%root)
os.system('rm -rf %s/raw/phase_velocity/*'%root)
os.system('rm -rf %s/pick_result/group_velocity/*'%root)
os.system('rm -rf %s/pick_result/phase_velocity/*'%root)
for file in os.listdir(root + '/test_result'):
    i += 1
    print(i, file)
    path = os.path.join(root + '/test_result', file)
    os.chdir(path)
    for group in glob.glob('newG*.txt'):
        name = group[5:]
        os.system('cp -rf ./%s %s/raw/group_velocity/%s'%(group, root, name))
    for phase in glob.glob('newC*.txt'):
        name = phase[5:]
        os.system('cp -rf ./%s %s/raw/phase_velocity/%s'%(phase, root, name))

os.chdir(root + '/raw/group_velocity')
all_ = 0
for file in glob.glob('*txt'):
    all_ += 1
    print(all_, file)
    num = 0
    data = np.loadtxt(file)[:, 1]
    for i in data:
        if i <= v_max and i >= v_min:
            num += 1
    if num >= points:
        # print(num)
        os.system('cp -rf %s %s/pick_result/group_velocity/'%(file, root))
        data2 = np.loadtxt(root + '/raw/phase_velocity/'+file)[:, 1]
        num2 = 0
        for i in data2:
            if i <= v_max and i >= v_min:
                num2 += 1
        if num2 >= points:
            # print(num2)
            os.system('cp -rf %s/raw/phase_velocity/%s %s/pick_result/phase_velocity/'%(root, file, root))
