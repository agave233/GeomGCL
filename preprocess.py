from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import csv
import pickle

MAX_ATOMIC_NUM = 100
BOND_FDIM = 14

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def featurize_mol_sdf(mol):
    pos_matrix = mol.GetConformer().GetPositions()
    atom_features = get_atom_features_sdf(mol,use_chirality=False)
    bond_features = dict()
    num_atoms = len(atom_features)
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        # print(begin_atom, end_atom)
        bond = mol.GetBondBetweenAtoms(begin_atom, end_atom)
        bond_features[(begin_atom, end_atom)] = get_bond_features(bond)
        # print(begin_atom, end_atom)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return num_atoms, atom_features, atom_types, pos_matrix, bond_features

def get_atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features_sdf(mol,use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                        use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                        use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):

    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                        'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if use_atom_symbol else []
        o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if use_degree else []
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]) if use_hybridization else []
        o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if use_implicit_valence else []
        o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if use_degree else []
        # o += [atom.GetProp("_GasteigerCharge")] if use_partial_charge else [] # some molecules return NaN
        o += [atom.GetIsAromatic()] if use_aromaticity else []
        o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                ring.IsAtomInRingOfSize(atom_idx, 4),
                ring.IsAtomInRingOfSize(atom_idx, 5),
                ring.IsAtomInRingOfSize(atom_idx, 6),
                ring.IsAtomInRingOfSize(atom_idx, 7),
                ring.IsAtomInRingOfSize(atom_idx, 8)] if use_ring_size else []
        o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if use_num_hydrogen else []

        if use_chirality:
            try:
                o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
            except:
                o += [False, False] + [atom.HasProp("_ChiralityPossible")]
        if use_hydrogen_bonding:
            o += [atom_idx in hydrogen_donor_match]
            o += [atom_idx in hydrogen_acceptor_match]
        if use_acid_base:
            o += [atom_idx in acidic_match]
            o += [atom_idx in basic_match]

        m.append(o)
    return np.array(m, dtype=float)


def get_bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def compute_2d_coords(mol, num_atoms):
    AllChem.Compute2DCoords(mol)
    return mol.GetConformer().GetPositions()[:, :2]

    info = Chem.MolToMolBlock(mol)
    coords = []
    for row in info.split('\n')[4:4+num_atoms]:
        x, y = list(map(float, row.split()[:2]))
        coords += [[x, y]]
    return coords

def generate_features_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    atom_features = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features.append(get_atom_features(atom))
        # atom_features.append(get_atom_features_sdf(atom))
    
    atom_features = [atom_features[i] for i in range(num_atoms)]
    atom_2dcoords = compute_2d_coords(mol, num_atoms)

    # Get bond features
    bond_features = dict()
    for a1 in range(num_atoms):
        for a2 in range(a1 + 1, num_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            f_bond = get_bond_features(bond)
            bond_features[(a1, a2)] = f_bond

    return num_atoms, atom_features, atom_2dcoords, bond_features


def get_smile_list_from_sdf(mols):
    smiles_list, y_list = [], []
    for idx, mol in tqdm(enumerate(mols)):
        if mol is None:
            print(idx, 'is None')
            continue
        smiles_list += [mol.GetProp('smile')]
        y_list += [float(mol.GetProp('target'))]
    
    return smiles_list, y_list

def get_smile_list_from_sdf(mols, name='smile', target_name=['target']):
    smiles_list, y_list = [], []
    for idx, mol in tqdm(enumerate(mols)):
        if mol is None:
            print(idx, 'is None')
            continue
        smiles_list += [mol.GetProp(name)]
        y_list += [[float(mol.GetProp(y_n)) for y_n in target_name]]
    
    return smiles_list, y_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='esol')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    mols = Chem.SDMolSupplier('%s/%s/%s.sdf' % (args.data_dir, args.dataset, args.dataset))
    if args.dataset in ['bbbp', 'hiv', 'bace', 'esol', 'lipop', 'freesolv']:
        x_y_name_dict = {'bbbp': ('smiles', 'p_np'), 'hiv': ('smiles', 'HIV_active'), 'bace': ('mol','Class'), 'esol': ('smiles','measured log solubility in mols per litre'), 'lipop': ('smiles','exp'), 'freesolv': ('smiles','expt')}
        name, target_name = x_y_name_dict[args.dataset]
        mol_list, logp_list = get_smile_list_from_sdf(mols)
        print('Number is', len(mol_list))
    else:
        if args.dataset in ['tox21', 'clintox']:
            x_y_name_dict = {'tox21':("smiles","NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53".split(',')), 'clintox':("smiles","FDA_APPROVED,CT_TOX".split(','))}
            name, target_name = x_y_name_dict[args.dataset]
        else:
            df = pd.read_csv('%s/%s/%s.csv' % (args.data_dir, args.dataset, args.dataset))
            name = df.columns[0]
            target_name = df.columns[1:]
        target_name = ['target-%d;'%(i+1) + target_name[i] for i in range(len(target_name))]
        mol_list, logp_list = get_smile_list_from_sdf(mols, name, target_name)
        print('Number is', len(mol_list))

    # 2D view
    data_mols = []
    for mol in tqdm(mol_list):
        data_mols.append(generate_features_from_smiles(mol))
    out_path = '%s/%s/%s_2d_processed.pkl' % (args.data_dir, args.dataset, args.dataset)
    with open(out_path, 'wb') as f:
        pickle.dump((data_mols, logp_list), f)
    
    # 3D view
    data_mols = []
    for mol in tqdm(mol_list):
        data_mols.append(featurize_mol_sdf(mol))
    out_path = '%s/%s/%s_3d_processed.pkl' % (args.data_dir, args.dataset, args.dataset)
    with open(out_path, 'wb') as f:
        pickle.dump((data_mols, logp_list), f)



