import os
import numpy as np
import sys
import copy

with open('atoms.txt') as f:
    read_data = f.readlines()
f.closed

for file in read_data:
    name = file.split('\n')[0]
    print(name)
    
    com1 = 'mkdir ' + name[0:-5]
    os.system(com1)
    
    com2 = 'cp calc_energy.py calc_inter_energy.py submitjob.pbs submitjob_int.pbs vdw_kernel.bindat ' + name[0:-5] + '/'
    os.system(com2)

    com3_1 = 'cd ' + name[0:-5] + '\n'
    com3_2 = 'sbatch submitjob.pbs ' + name + '\n'
    com3_3 = 'mkdir interaction' + '\n'
    com3_4 = 'mv calc_inter_energy.py submitjob_int.pbs interaction/' + '\n'
    com3_5 = 'cd interaction' + '\n'
    com3_6 = 'sbatch submitjob_int.pbs ' + name
    com3 = com3_1 + com3_2 + com3_3 + com3_4 + com3_5 + com3_6
    os.system(com3)