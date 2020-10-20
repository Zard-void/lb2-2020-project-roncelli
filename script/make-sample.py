import random
import pandas as pd
import os

from Bio.PDB import PDBList
from tqdm.notebook import tqdm
from sultan.api import Sultan


def extract_chain(in_path, out_path, chain):
    base_name = os.path.basename(in_path)
    name = os.path.splitext(base_name)[0]
    extension = '.cif'
    with open(in_path + extension, 'r') as file, open(out_path + name + '_' + chain + extension, 'a') as out_file:
        if os.path.exists(out_path + name + '_' + chain + extension):
            os.remove(out_path + name + '_' + chain + extension)
        print(file.readline().rstrip() + '_' + chain + '\n#' + '\nloop_', end='\n', file=out_file)
        counter = 0
        chain_index = 0
        for line in file:
            if line.startswith('_atom_site.'):
                while line.startswith('_atom_site.'):
                    if 'auth_asym_id' in line:
                        chain_index = counter
                    print(line, end='', file=out_file)
                    line = file.readline()
                    counter += 1
            line2 = line.split()
            if len(line2) >= chain_index + 1 and line2[0] == 'ATOM' and line2[chain_index] == chain:
                print(line, end='', file=out_file)
        print('#', file=out_file)


def download_structures(structure_list):
    pdbl = PDBList()
    for record in tqdm(structure_list):
        structure_id = record[:4]
        chain = record[5:]
        pdbl.retrieve_pdb_file(structure_id, pdir = '.', file_format='mmCif')
        os.rename(structure_id.lower() + '.cif', structure_id + '.cif')

extract_chain(structure_id.lower(), 'singlechains/', chain)

def ss_mapper(structure):
    mapped_structure = []
    position = int()
    for secondary in structure:
        if secondary == ' ' or secondary == 'T' or secondary == 'S' or secondary == '-':
             secondary = 'C'
        elif secondary == 'H' or secondary == 'G' or secondary == 'I':
            secondary = 'H'
        elif secondary == 'B' or secondary == 'E':
            secondary = 'E'
        else:
            if secondary == '\n':
                raise Exception('Character is a newline. Did you remember to rstrip?')
            else:
                raise Exception('Aminoacid in position ' + str(position) + ' is ' + "'" + secondary + "'")
        position += 1
        mapped_structure.append(secondary)
    mapped_structure = ''.join([str(elem) for elem in mapped_structure])
    return mapped_structure

def run_dssp(structure_list, out_path):
    for pdb_id in tqdm(structure_list):
        with Sultan.load(sudo=False) as s:
            s.dssp('-i ' + pdb_id.lower() + '.cif ' +
                   '-o ' + out_path + pdb_id + '.dssp')


def convert_dssp_to_df(structure_list):
    dssp_df = pd.DataFrame(columns=['PDB ID', "Sequence", "Secondary structure"])
    for pdb_id in structure_list:
        if pdb_id + '.dssp' in os.listdir('.'):
            with open(pdb_id + '.dssp') as handle:
                sequence = list()
                structure = list()
                line = handle.readline().split()
                while line[0] != '#':
                    line = handle.readline().split()
                for line in handle:
                    line = list(line)
                    if line[13] == '!':
                        sequence.append('X')
                        structure.append(' ')
                    else:
                        sequence.append(line[13])
                        structure.append(line[16])
            sequence = ''.join(sequence)
            structure = ''.join(structure)
            structure = ss_mapper(structure)
            dssp_df = dssp_df.append({'PDB ID': pdb_id,
                                      'Sequence': sequence,
                                      'Secondary structure': structure}, ignore_index=True)
    return dssp_df

os.chdir('../data/test')
representatives = pd.read_csv('full_test.tsv', sep='\t', index_col=0)
test_list = random.sample(representatives['ID'].to_list(), k=150)
