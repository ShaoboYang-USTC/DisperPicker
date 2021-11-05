import os
import glob

i = 0
root = os.path.abspath(os.path.dirname(__file__))
print(root)
os.system('rm -rf %s/plot1/*'%root)
os.system('rm -rf %s/plot2/*'%root)
for file in os.listdir(root + '/test_result/'):
    i += 1
    print(i,file)
    path = os.path.join(root + '/test_result/', file)
    os.chdir(path)
    for fig in glob.glob('*png'):
        os.system('cp -rf %s %s/plot1/'%(fig, root))
    for fig in glob.glob('*jpg'):
        os.system('cp -rf %s %s/plot2/'%(fig, root))
