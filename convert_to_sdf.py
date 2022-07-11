import csv
import argparse
import numpy as np
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem


def load_csv(path, x_name, y_name):
    x, y = [], []

    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(row[x_name])
            y.append(float(row[y_name]))

    x = np.array(x, dtype=str)
    y = np.array(y, dtype=float)

    return x, y


def optimize_conformer(idx, smi, m, prop, algo="MMFF"):
    print("Calculating {}: {} ...".format(idx, Chem.MolToSmiles(m)))

    mol = Chem.AddHs(m)

    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if k != 0:
            return None, None, None

    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    mol = Chem.RemoveHs(mol)
    return smi, mol, prop


def random_rotation_matrix():
    theta = np.random.rand()
    r_x = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand()
    r_y = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand()
    r_z = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def rotate_molecule(path, target_path, count=10):
    # Load dataset
    mols = Chem.SDMolSupplier(path)
    rotated_mols = []

    print("Loaded {} Molecules from {}".format(len(mols), path))

    print("Rotating Molecules...")
    for mol in mols:
        for _ in range(count):
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()

                pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
                pos_rotated = np.matmul(random_rotation_matrix(), pos)

                mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)

            rotated_mols.append(mol)

    w = Chem.SDWriter(target_path)
    for m in rotated_mols:
        if m is not None:
            w.write(m)
    print("Saved {} Molecules to {}".format(len(rotated_mols), target_path))


def converter(path, target_path, dataset, algo, process=20, debug=False):
    # Load dataset
    print("Loading Dataset...")
    x_y_name_dict = {'bace': ('mol','Class'), 'esol': ('smiles','measured log solubility in mols per litre'), 'lipop': ('smiles','exp'), 'freesolv': ('smiles','expt')}
    name, target_name = x_y_name_dict[dataset]
    if ".csv" in path:
        x, y = load_csv(path, name, target_name)
        smis, mols, props = [], [], []
        for i in range(len(x)):
            smi, prop = x[i], y[i]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smis.append(smi)
                mols.append(mol)
                props.append(prop)

        mol_idx = list(range(len(mols)))
        algo = [algo]*len(mols)

    else:
        raise ValueError("Unsupported file type.")
    print("Loaded {} Molecules from {}".format(len(mols), path))

    # Optimize coordinate using multiprocessing
    print("Optimizing Conformers...")
    pool = mp.Pool(process)
    results = pool.starmap(optimize_conformer, zip(mol_idx, smis, mols, props, algo))

    # Collect results
    smi_list, mol_list, prop_list = [], [], []
    for smi, mol, prop in results:
        smi_list.append(smi)
        mol_list.append(mol)
        prop_list.append(prop)

    # Remove None and add properties
    mol_list_filtered = []
    for smi, mol, prop in zip(smi_list, mol_list, prop_list):
        if mol is not None:
            mol.SetProp("smile", str(smi))
            mol.SetProp("target", str(prop))
            mol_list_filtered.append(mol)
        else:
            print('error')
    print("{} Molecules Optimized".format(len(mol_list_filtered)))

    # Save molecules
    print("Saving File...")
    w = Chem.SDWriter(target_path)
    for m in mol_list_filtered:
        w.write(m)
    print("Saved {} Molecules to {}".format(len(mol_list_filtered), target_path))

# run: python convert_to_sdf.py --dataset esol --process 2 --algo MMFF
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=int, default=20)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--algo', type=str, default='MMFF') # ETKDG, UFF, MMFF
    args = parser.parse_args()

    converter("./data/%s/%s.csv"%args.dataset, "./data/%s/%s.sdf"%(args.dataset, args.dataset), args.dataset, args.algo, process=args.process)
