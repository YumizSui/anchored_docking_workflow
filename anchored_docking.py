import os
import shutil
import json
import copy
import argparse
import numpy as np
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

def base36encode(number, length=2):
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result = ''
    while number > 0:
        number, remainder = divmod(number, 36)
        result = chars[remainder] + result
    return result.zfill(length)

def calculate_center_of_mass(mol):
    positions = mol.GetConformer().GetPositions()
    masses = [atom.GetMass() for atom in mol.GetAtoms()]
    total_mass = sum(masses)
    weighted_positions = positions * np.array(masses)[:, np.newaxis]
    center_of_mass = np.sum(weighted_positions, axis=0) / total_mass
    return center_of_mass

def get_atype(atom, atom_types):
    mol = atom.GetOwningMol()
    for type_info in atom_types:
        pattern = Chem.MolFromSmarts(type_info['smarts'])
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                if atom.GetIdx() in match:
                    return type_info['atype']
    return None

def find_equivalent_atoms(mol, atom1_idx, atom2_idx):
    Chem.SanitizeMol(mol, Chem.SANITIZE_PROPERTIES)
    symmetry_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    return symmetry_classes[atom1_idx] == symmetry_classes[atom2_idx]

def find_anchors(smart_pattern):
    pattern_mol = Chem.MolFromSmarts(smart_pattern)
    heavy_atoms = [atom.GetIdx() for atom in pattern_mol.GetAtoms() if atom.GetAtomicNum() > 1]
    anchors = []
    for idx in heavy_atoms:
        if all(not find_equivalent_atoms(pattern_mol, idx, idx2) for idx2 in heavy_atoms if idx != idx2):
            anchors.append((smart_pattern, idx))
    return anchors

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file", help="Input SDF file", required=True)
    parser.add_argument("--input_file", help="Input SDF file", required=True)
    parser.add_argument("--receptor_file", help="Receptor PDB file", required=True)
    parser.add_argument("--docking_dir", help="Directory for docking", required=True)
    parser.add_argument("--query_smarts", help="Manual SMARTS query for anchor points", default=None)
    parser.add_argument("--anchor_indices", nargs="+", type=int, help="Specify anchor points by atom indices. Query SMILES must be provided. Indices is 0-based.", default=None)
    parser.add_argument("--anchor_num", type=int, default=3, help="Number of anchor points")
    parser.add_argument("--anchor_mode", choices=["random", "area"], default="random", help="Anchor point selection mode. the number of anchor of area mode is always 3.")
    parser.add_argument("--size", type=float, help="Size of docking box", default=20.0)
    parser.add_argument("--force", action="store_true", help="Force to overwrite the directory")
    parser.add_argument("--adg_path", default="autodock-gpu")
    parser.add_argument("--adfr_path", default="ADFRsuite-1.0")
    parser.add_argument("--scripts_path", default="scripts")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--clean", action="store_true", help="Clean the directory after docking")
    args = parser.parse_args()

    if args.anchor_indices is not None and args.query_smarts is None:
        raise ValueError("Query SMILES must be provided when anchor indices are specified.")
    if args.anchor_indices is not None and args.anchor_num > 0:
        print("Anchor indices are specified. Anchor number will be ignored.")
    if args.anchor_indices is not None and args.anchor_mode == "area":
        print("Anchor indices are specified. Anchor mode will be ignored.")
    return args

def prepare_docking_directory(docking_dir, force):
    if force:
        if os.path.exists(docking_dir):
            shutil.rmtree(docking_dir)
        os.makedirs(docking_dir)
    else:
        if not os.path.exists(docking_dir):
            os.makedirs(docking_dir)
        else:
            raise ValueError(f"Directory '{docking_dir}' already exists. Use --force to overwrite.")

def prepare_boxsize_file(docking_dir, size, center_of_mass):
    grid_dict = {
        "size_x": size,
        "size_y": size,
        "size_z": size,
        "center_x": center_of_mass[0],
        "center_y": center_of_mass[1],
        "center_z": center_of_mass[2],
    }
    grid_text = "\n".join(f"{key} = {value}" for key, value in grid_dict.items())
    with open(os.path.join(docking_dir, "boxsize.txt"), "w") as f:
        f.write(grid_text)

def find_anchor_points(template_mol, input_mol, anchor_num, samples, random_seed, query_smarts=None, anchor_indices=None, anchor_mode="random", verbose=False):
    if verbose:
        print("Template_SMILES:", Chem.MolToSmiles(template_mol))
        print("Input_SMILES:", Chem.MolToSmiles(input_mol))
    if query_smarts is not None:
        pattern_smiles = query_smarts
        pattern_smarts = query_smarts
        pattern_mol = Chem.MolFromSmiles(pattern_smiles)
        smart_pattern_mol = Chem.MolFromSmarts(pattern_smiles)
        ptn2template_mapping = {
            i: j
            for i, j in zip(
                pattern_mol.GetSubstructMatch(smart_pattern_mol),
                template_mol.GetSubstructMatch(smart_pattern_mol),
            )
        }
    else:
        mcs_res = rdFMCS.FindMCS(
            [input_mol, template_mol],
            atomCompare=rdFMCS.AtomCompare.CompareAny,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringCompare=rdFMCS.RingCompare.StrictRingFusion,
            matchValences=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchChiralTag=False,
        )
        substructure = input_mol.GetSubstructMatch(mcs_res.queryMol)
        pattern_smiles = Chem.MolFragmentToSmiles(input_mol, substructure)
        pattern_smarts = Chem.MolToSmarts(Chem.MolFromSmiles(pattern_smiles))
        pattern_mol = Chem.MolFromSmiles(pattern_smiles)
        smart_pattern_mol = Chem.MolFromSmarts(pattern_smarts)
        input2template_mapping = {
            i: j
            for i, j in zip(
                input_mol.GetSubstructMatch(mcs_res.queryMol),
                template_mol.GetSubstructMatch(mcs_res.queryMol),
            )
        }
        ptn2input_mapping = {
            i: j
            for i, j in zip(
                pattern_mol.GetSubstructMatch(smart_pattern_mol),
                input_mol.GetSubstructMatch(smart_pattern_mol),
            )
        }
        ptn2template_mapping = {i: input2template_mapping[j] for i, j in ptn2input_mapping.items()}
    if verbose:
        print("Pattern SMILES:", pattern_smiles)
        print("Pattern SMARTS:", pattern_smarts)

    anchor_patterns_all = find_anchors(pattern_smarts)
    anchor_positions_all = [
        list(template_mol.GetConformer().GetAtomPosition(ptn2template_mapping[idx]))
        for _, idx in anchor_patterns_all
    ]

    if anchor_indices is not None:
        avail_anchors = [idx for anchor,idx in anchor_patterns_all]
        if not all([idx in avail_anchors for idx in anchor_indices]):
            raise ValueError("Some anchor indices are not found in the anchor points, Available anchor indices are: ", avail_anchors)
        anchor_patterns = [(pattern_smarts, idx) for idx in anchor_indices]
        anchor_positions = [
            list(template_mol.GetConformer().GetAtomPosition(ptn2template_mapping[idx]))
            for idx in anchor_indices
        ]
    else:
        if anchor_num > 0:
            rng = np.random.default_rng(random_seed)
            if anchor_mode == "random":
                if len(anchor_patterns_all) < anchor_num:
                    raise ValueError("Not enough anchor points found.")
                sample_idx = rng.choice(len(anchor_patterns_all), size=min(anchor_num, len(anchor_patterns_all)), replace=False)
                anchor_patterns = [anchor_patterns_all[idx] for idx in sample_idx]
                anchor_positions = [anchor_positions_all[idx] for idx in sample_idx]
            elif anchor_mode == "area":
                anchor_num = 3
                area_max = -1
                anchor_patterns = None
                anchor_positions = None
                for _ in range(samples):
                    if len(anchor_patterns_all) < anchor_num:
                        raise ValueError("Not enough anchor points found.")
                    sample_idx = rng.choice(len(anchor_patterns_all), size=anchor_num, replace=False)
                    positions = [np.array(anchor_positions_all[idx]) for idx in sample_idx]
                    if len(positions) != 3:
                        raise ValueError("Not enough anchor points found.")
                    area = 0.5 * np.linalg.norm(
                        np.cross(positions[1] - positions[0], positions[2] - positions[0])
                    )
                    if area > area_max:
                        area_max = area
                        anchor_patterns = [anchor_patterns_all[idx] for idx in sample_idx]
                        anchor_positions = [anchor_positions_all[idx] for idx in sample_idx]
            else:
                raise ValueError(f"Unknown anchor mode: {anchor_mode}")
        else:
            anchor_patterns = anchor_patterns_all
            anchor_positions = anchor_positions_all
    return anchor_patterns, anchor_positions, pattern_mol


def generate_parameters_json(scripts_path, anchor_patterns, pattern_mol):
    with open(os.path.join(scripts_path, "base_parameters.json")) as f:
        base_parameters = json.load(f)
    atom_types = base_parameters['ATOM_PARAMS']['ad4']
    parameters = copy.deepcopy(base_parameters)

    new_atype2atype = {}
    anchor_dict_list = []

    for i, (smarts, idx) in enumerate(anchor_patterns):
        atom = pattern_mol.GetAtomWithIdx(idx)
        atype = get_atype(atom, atom_types)
        if atype is None:
            raise ValueError(f"Could not determine atom type for atom index {idx}.")
        new_atype = 'A' + base36encode(i, 2)
        d = {"smarts": smarts, "IDX": [idx + 1], "atype": new_atype}
        parameters['ATOM_PARAMS']['ad4'].append(d)
        anchor_dict_list.append(d)
        new_atype2atype[new_atype] = atype
    return parameters, anchor_dict_list, new_atype2atype

def prepare_ligand(docking_dir, input_mol):
    inputmol_h = Chem.AddHs(input_mol, addCoords=True)
    AllChem.EmbedMolecule(inputmol_h, AllChem.ETKDGv3())
    ligand_sdf = os.path.join(docking_dir, "ligand.sdf")
    with Chem.SDWriter(ligand_sdf) as f:
        f.write(inputmol_h)
    subprocess.run(
        ["mk_prepare_ligand.py", "-p", "parameters.json", "-i", "ligand.sdf", "-o", "ligand.pdbqt"],
        cwd=docking_dir,
        check=True
    )

def prepare_receptor(docking_dir, receptor_file, adfr_path):
    receptor_pdb = os.path.join(docking_dir, "receptor.pdb")
    shutil.copy(receptor_file, receptor_pdb)
    prepare_receptor_cmd = [
        os.path.join(adfr_path, "bin", "prepare_receptor"),
        "-r",
        "receptor.pdb",
        "-o",
        "receptor.pdbqt",
    ]
    subprocess.run(prepare_receptor_cmd, cwd=docking_dir, check=True)

def prepare_grid(docking_dir, scripts_path, adg_path):
    write_gpf_cmd = [
        os.path.join(scripts_path, "write-gpf.py"),
        "receptor.pdbqt",
        "-b",
        "boxsize.txt",
        "--mapprefix",
        "rec",
    ]
    subprocess.run(write_gpf_cmd, cwd=docking_dir, check=True)
    autogrid_cmd = [os.path.join(adg_path, "autogrid4"), "-p", "rec.gpf", "-l", "rec.glg"]
    subprocess.run(autogrid_cmd, cwd=docking_dir, check=True)

def add_bias_to_map_file(docking_dir, scripts_path, anchor_dict_list, anchor_positions, new_atype2atype):
    for anchor_d, position in zip(anchor_dict_list, anchor_positions):
        atype = anchor_d["atype"]
        old_atype = new_atype2atype[atype]
        addbias_cmd = [
            os.path.join(scripts_path, "addbias.py"),
            "-i",
            f"rec.{old_atype}.map",
            "-o",
            f"rec.{atype}.map",
            "-x",
            str(position[0]),
            str(position[1]),
            str(position[2]),
        ]
        subprocess.run(addbias_cmd, cwd=docking_dir, check=True)
        insert_type_cmd = [
            os.path.join(scripts_path, "insert_type_in_fld.py"),
            "rec.maps.fld",
            "--newtype",
            atype,
        ]
        subprocess.run(insert_type_cmd, cwd=docking_dir, check=True)

def run_docking(docking_dir, adg_path, anchor_dict_list, new_atype2atype):
    t_options_map = {}
    for anchor_d in anchor_dict_list:
        atype = anchor_d["atype"]
        old_atype = new_atype2atype[atype]
        t_options_map.setdefault(old_atype, []).append(atype)
    t_options = "/".join([f"{','.join(v)}={k}" for k, v in t_options_map.items()])

    adgpu_cmd = [
        os.path.join(adg_path, "adgpu"),
        "-L",
        "ligand.pdbqt",
        "-M",
        "rec.maps.fld",
        "-T",
        t_options,
    ]
    subprocess.run(adgpu_cmd, cwd=docking_dir, check=True)

def export_results(docking_dir):
    export_cmd = ["mk_export.py", "ligand.dlg", "-o", "ligand_docked.sdf"]
    subprocess.run(export_cmd, cwd=docking_dir, check=True)

def clean_directory(docking_dir):
    for f in os.listdir(docking_dir):
        if f != "ligand_docked.sdf":
            os.remove(os.path.join(docking_dir, f))

def main():
    args = parse_arguments()

    adg_path = os.path.abspath(args.adg_path)
    adfr_path = os.path.abspath(args.adfr_path)
    scripts_path = os.path.abspath(args.scripts_path)

    # Prepare docking directory
    prepare_docking_directory(args.docking_dir, args.force)

    # Load molecules
    template_mol = Chem.SDMolSupplier(args.template_file)[0]
    input_mols = list(Chem.SDMolSupplier(args.input_file))
    if template_mol is None or input_mols is None:
        raise ValueError("Failed to load molecule files.")

    for index, input_mol in enumerate(input_mols):
        docking_dir = os.path.join(args.docking_dir, str(index))
        os.makedirs(docking_dir)

        # Calculate center of mass for the grid center
        center_of_mass = calculate_center_of_mass(template_mol)

        # Prepare boxsize.txt
        prepare_boxsize_file(docking_dir, args.size, center_of_mass)

        # Find anchor points
        anchor_patterns, anchor_positions, pattern_mol = find_anchor_points(
            template_mol,
            input_mol,
            args.anchor_num,
            args.samples,
            args.random_seed,
            args.query_smarts,
            args.anchor_indices,
            args.anchor_mode,
            args.verbose,
        )
        with open(os.path.join(docking_dir, "anchor_positions.json"), "w") as f:
            json.dump(anchor_positions, f, indent=4)


        # Generate parameters.json
        parameters, anchor_dict_list, new_atype2atype = generate_parameters_json(
            scripts_path,
            anchor_patterns,
            pattern_mol,
        )

        # Save parameters.json
        with open(os.path.join(docking_dir, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=4)

        # Prepare ligand.pdbqt
        prepare_ligand(docking_dir, input_mol)

        # Prepare receptor.pdbqt
        prepare_receptor(docking_dir, args.receptor_file, adfr_path)

        # Prepare grid
        prepare_grid(docking_dir, scripts_path, adg_path)

        # Add bias to the map file
        add_bias_to_map_file(
            docking_dir,
            scripts_path,
            anchor_dict_list,
            anchor_positions,
            new_atype2atype,
        )

        # Run docking
        run_docking(docking_dir, adg_path, anchor_dict_list, new_atype2atype)

        # Export results
        export_results(docking_dir)

        # Clean up if needed
        if args.clean:
            clean_directory(docking_dir)

if __name__ == "__main__":
    main()
