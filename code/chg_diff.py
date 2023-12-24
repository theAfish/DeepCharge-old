from deepmd.infer import DeepPot
import numpy as np
import ase.io
import pandas as pd
import asyncio
from deepmd.calculator import DP

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#     return wrapped


a2n = {'Si':0, 'X':1}
# a2n = {'H':0, 'O':1, 'X':2}
# a2n = {'Ti':0, 'O':1, 'X':3}
prob_id = 1
num_prob_per_frame = 200

dp = DeepPot("model/si.pb")

chg = pd.read_csv("output/si/si-82.xyz", skiprows=0, delimiter=' ', header=None, skipinitialspace=True)
pos = chg[[0,1,2]]
num_probs = chg.shape[0]

atoms = ase.io.read("remote_test/si/82/POSCAR")
coord = atoms.get_positions() #.reshape([1, -1])
cell = atoms.get_cell().reshape([1, -1])
atype = atoms.get_chemical_symbols()
atype = [a2n[i] for i in atype]
atype_p = np.append(atype, [prob_id] * num_prob_per_frame)
num_atoms = atoms.get_global_number_of_atoms()


out = []


for i in range(0, num_probs, num_prob_per_frame):
    probs = pos.iloc[i:i+num_prob_per_frame]
    if probs.shape[0] == num_prob_per_frame:
        coord_with_prob = np.append(coord, probs, axis=0)
        atype_with_prob = atype_p
    else:
        coord_with_prob = np.append(coord, probs, axis=0)
        atype_with_prob = np.append(atype, [prob_id]*probs.shape[0])

    result = dp.eval(coord_with_prob.reshape([1,-1]), cell, atype_with_prob, atomic=True)

    chg.loc[i:i+num_prob_per_frame-1, 4] = result[3][0][num_atoms:]
    # print("{} finished, {} remained".format(i, num_probs-i))


chg.to_csv('output/si/si-82-diff.xyz', sep=' ', index=False)
print('finished')