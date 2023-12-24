import numpy as np
import pandas as pd
import dpdata
import subprocess
import os
import shutil

atom2num = {"H":1, "C":6, "O":8, "F":9,"Si":14,"Fe":26, "I":53, "Cs":55, "Pb":82, "Ti":22}

N_PROBS = 200
N_FRAMES = 500
PROB_ID = 1



#######      Grab info from OUTCAR      ########
def outcar2set(input_dir, outcar_name, output_dir, type_map):
    d_outcar = dpdata.LabeledSystem(input_dir + outcar_name, type_map=type_map)
    print(d_outcar)
    d_outcar.to("deepmd/npy", output_dir + "dpmd_npy", )
    atomic_num = [atom2num[i] for i in d_outcar.get_atom_names()] + ["\n"]
    atomic_num = [str(element) for element in atomic_num]
    return ' '.join(atomic_num), d_outcar.get_natoms()

#######      Convert CHGCAR 2 xyz      #######
def chgcar2xyz(atomic_num, input_dir, chgcar_name, output_name='density.xyz'):
    with subprocess.Popen(['perl', 'chg2cube.pl', input_dir+chgcar_name, output_dir+"density.cube"], stdin=subprocess.PIPE) as proc:
        proc.stdin.write(b'1\n')
        proc.stdin.write(atomic_num.encode())
        # proc.stdin.write('1 6'.encode())
        proc.communicate()
    subprocess.call(['python', 'cube2xyz.py', '-f', output_dir+'density.cube', '-o', output_dir+output_name])



#######     Preparing for dataset       #######
def create_path(output_dir, N_PROBS=N_PROBS, PROB_ID=PROB_ID, dataset_name="sys.000", set_num="000"):
    if not os.path.exists(output_dir+dataset_name):
        os.makedirs(output_dir+dataset_name+"/set."+set_num)

        # create type.raw and type_map.raw
        shutil.copy(output_dir+"dpmd_npy/type.raw", output_dir+dataset_name+"/")
        shutil.copy(output_dir+"dpmd_npy/type_map.raw", output_dir+dataset_name+"/")
        with open(output_dir+dataset_name+"/type_map.raw", 'a') as file:
            file.write('Prob')
        with open(output_dir+dataset_name+"/type.raw", 'a') as file:
            for i in range(N_PROBS):
                file.write('{}\n'.format(PROB_ID))
    else:
        os.makedirs(output_dir +dataset_name+"/set." + set_num)


#######     Seperating & concatenate    #######
def set_dataset(num_atoms, N_PROBS=N_PROBS, N_FRAMES=N_FRAMES, dataset_name="sys.000", set_num="000", charge_xyz="density.xyz"):
    df = pd.read_csv(output_dir+charge_xyz, sep=' ', header=None, names=['x', 'y', 'z', 'c'])

    # uncomment for high density dataset
    # df['c'] = df['c'] - df0['c']
    # df = df[abs(df['c']) > 0.1]
    # print(df.describe())

    num_lines = len(df)
    o_box = np.empty([0,9])
    o_coord = np.empty([0,(N_PROBS+num_atoms)*3])
    o_atom_ener = np.empty([0, N_PROBS+num_atoms])
    # for i in range(df.count()['c'] // N_PROBS):
    for i in range(N_FRAMES):
        probs = df.sample(N_PROBS)
        df = df.drop(probs.index)
        o_box = np.append(o_box, a_box, axis=0)
        _coord = np.append(a_coord, probs.values[:, :3].flatten())
        o_coord = np.append(o_coord, [_coord], axis=0)
        _atom_ener = np.append([0] * num_atoms, probs.values[:, -1].flatten())
        o_atom_ener = np.append(o_atom_ener, [_atom_ener], axis=0)
        if i%100 == 0:
            print("Current frame: {}".format(i))
    o_ener = o_atom_ener.sum(axis=1)
    np.save(output_dir + dataset_name +"/set."+set_num+"/box.npy", o_box)
    np.save(output_dir + dataset_name +"/set."+set_num+"/coord.npy", o_coord)
    np.save(output_dir + dataset_name +"/set."+set_num+"/atom_ener.npy", o_atom_ener)
    np.save(output_dir + dataset_name +"/set."+set_num+"/energy.npy", o_ener)
    print("There are {} probs left".format(len(df)))



def clean():
    shutil.rmtree(output_dir + "dpmd_npy")
    os.remove(output_dir + "density.cube")

for i in range(101):
    label = i+10
    input_dir = "remote_test/fe/{}/".format(label)
    type_map = ["Fe"]
    output_dir = "output/fe/"
    outcar_name = "OUTCAR"
    chgcar_name = "CHGCAR_mag"
    set_num = "00{}".format(label)


    atomic_nums, num_atoms = outcar2set(input_dir, outcar_name, output_dir, type_map=type_map)
    #######     Reading the atomic info     #######
    a_box = np.load(output_dir + "dpmd_npy/set.000/box.npy")
    a_coord = np.load(output_dir + "dpmd_npy/set.000/coord.npy")

    # if CHARGED:
    #     a_coord = np.append(a_coord, [chg_pos], axis=1)


    chgcar2xyz(atomic_nums, input_dir, chgcar_name=chgcar_name, output_name='fe-{}.xyz'.format(label))
    create_path(output_dir, dataset_name="sys.000", set_num=set_num)

    set_dataset(num_atoms, charge_xyz='fe-{}.xyz'.format(label), dataset_name="sys.000", set_num=set_num)

    clean()
    print('finish {}'.format(label))

