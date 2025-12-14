import os

import ase
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io import read
import numpy as np
from ptagnn2.calculators import MACECalculator
from ase.data import covalent_radii,chemical_symbols
from ase import build
import sys

def writeeffectmij(descriptors_mji, sender, receiver, length,atomic_numbers,writefile):
    for mji,s,r,l in zip(descriptors_mji, sender, receiver, length):
        if os.path.exists(writefile):
            with open(writefile,'a') as f:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(atomic_numbers[s],atomic_numbers[r],s,r,l.item(),mji))
        else:
            with open(writefile,'w') as f:
                f.write("# type_I\ttype_j\tid_I\tid_j\tr_Ij\tE_Ij\n")
                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(atomic_numbers[s],atomic_numbers[r],s,r,l.item(),mji))

if len(sys.argv) < 3:
    print("Usage: python script.py modelpath xyzpath")
else:
    calculator = MACECalculator(model_paths=sys.argv[1], device='cpu')
    configpath = sys.argv[2]
    draw_bond = True
    configs = os.listdir(configpath)
    for config in configs:
        if ".xyz" in config:
            configfile = os.path.join(configpath,config)
            print(configfile)
            dirpath = configfile.split(".")[0]
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            config_name = configfile.split("\\")[-1].split(".")[0]
            init_confs = ase.io.read(configfile, index=":")
            write_count = 0
            for j,init_conf in enumerate(init_confs):
                print(j,"/",len(init_confs))
                descriptors_mji, sender, receiver, length = calculator.get_effectivemji(init_conf)
                writefile = os.path.join(dirpath, str(write_count) + ".txt")
                if os.path.exists(writefile):
                    pass
                else:
                    writeeffectmij(descriptors_mji, sender, receiver, length, init_conf.numbers, writefile)
                write_count = write_count+1