import os
from contextlib import ExitStack
from tempfile import NamedTemporaryFile, TemporaryDirectory
import biotite.application.dssp as dssp
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import pandas as pd
from Bio import SeqIO
from Bio.Blast.Applications import NcbipsiblastCommandline
from sultan.api import Sultan
from tqdm import tqdm


def filter_entries(pdb_download):
    pdb_download = pdb_download[pdb_download['Sequence'].str.contains('Sequence') == False]
    pdb_download = pdb_download[pdb_download['Sequence'].str.contains('X') == False]
    pdb_download['Chain Length'] = pdb_download['Chain Length'].astype(int)
    pdb_download = pdb_download[pdb_download['Chain Length'].isna() == False]
    pdb_download = pdb_download.reset_index(drop=True)
    return pdb_download


def reduce_redundancy(pdb_download):
    with Sultan.load(sudo=False) as s:
        jpred_fasta = '../data/training/jpred.fasta'
        with ExitStack() as stack, \
                NamedTemporaryFile(mode='w+t') as unclustfast, \
                TemporaryDirectory() as tmpdir:
            s.cp(f'{jpred_fasta} {tmpdir}')
            tmp_list = [stack.enter_context(NamedTemporaryFile()) for x in range(5)]
            clustdb, repr, fastarepr, similarrepr, cluster = tmp_list
            for entry in pdb_download.index:
                sequence = pdb_download.loc[entry, 'Sequence']
                pdb_id = pdb_download.loc[entry, 'PDB ID']
                chain = pdb_download.loc[entry, 'Chain ID'].split(',')[0]
                print(f'>{pdb_id}_{chain}', file=unclustfast)
                print(f'{sequence}', file=unclustfast)
            s.mmseqs(f'createdb {unclustfast.name} {clustdb.name}').run()
            s.mmseqs(f'cluster {clustdb.name} {cluster.name} {tmpdir} --min-seq-id 0.3 -c 0.5 --cov-mode 0').run()
            s.mmseqs(f'createsubdb {cluster.name} {clustdb.name} {repr.name}').run()
            s.mmseqs(f'convert2fasta {repr.name} {fastarepr.name}').run()
            s.makeblastdb(f'-in {jpred_fasta} -dbtype prot -out {tmpdir}/jpred.fasta').run()
            s.blastp(f'-query {fastarepr.name} -out {similarrepr.name} -db {tmpdir}/jpred.fasta '
                     '-outfmt 6 '
                     '-evalue 0.01').run()

            representatives = list()
            for record in SeqIO.parse(f'{fastarepr.name}', 'fasta'):
                representatives.append([record.id, str(record.seq)])
            representatives = pd.DataFrame(representatives, columns=['ID', 'Sequence'])
            representatives_similar = pd.read_csv(f'{similarrepr.name}',
                                                  sep='\t',
                                                  usecols=[0, 2],
                                                  names=['Query', 'Identity']
                                                  )

            similar_id = representatives_similar[representatives_similar['Identity'] > 30]
            similar_id = similar_id['Query'].values.tolist()
            test_set = representatives[~representatives['ID'].isin(similar_id)]
    return test_set


if __name__ == '__main__':
    pdb_raw_download = pd.read_csv('../data/test/pdb_raw_download.csv', usecols=[0, 1, 2, 3, 4, 5], header=0)
    pdb_raw_download = filter_entries(pdb_raw_download)
    full_test = reduce_redundancy(pdb_raw_download)
    full_test.to_csv('../data/test/full_test.tsv', sep='\t')
    full_test[['ID', 'Chain']] = full_test['ID'].str.split("_", expand=True)

    new_df = pd.DataFrame()
    list_h = ['H']
    list_e = ['E', 'B']

    for _, sample in tqdm(full_test.iterrows(), total=len(full_test)):
        sample_id = sample['ID']
        chain = sample['Chain']
        sequence = sample['Sequence']
        # file_name = rcsb.fetch(sample_id, "pdbx", '../data/test/pdb', overwrite=False)
        pdbx_file = pdbx.PDBxFile.read(f'../data/test/pdb/{sample_id}.pdbx')
        array = pdbx.get_structure(pdbx_file, model=1, )
        structure = array[struc.filter_amino_acids(array)]
        structure = structure[structure.chain_id == chain]
        structure.set_annotation('chain_id', ['A' for el in structure.chain_id])
        sse = dssp.DsspApp.annotate_sse(structure)
        sse = ['H' if ss in list_h else 'E' if ss in list_e else 'C' for ss in sse]
        sse = ''.join(sse)
        if len(sse) != len(sequence):
            continue
        sample['Structure'] = sse
        new_df = new_df.append(sample)

    for _, sample in tqdm(new_df.iterrows(), total=len(new_df)):
        with NamedTemporaryFile(mode='w') as blast_in:
            sample_id = sample['ID']
            chain = sample['Chain']
            out_name = sample_id + '_' + chain
            sequence = sample['Sequence']
            print(f'>{sample_id}_{chain}', file=blast_in)
            print(sequence, file=blast_in)
            blast_name = blast_in.name
            blast_in.seek(0)
            cmd = NcbipsiblastCommandline(query=blast_name,
                                          db='../data/swiss-prot/uniprot_sprot.fasta',
                                          evalue=0.01,
                                          num_iterations=3,
                                          out_ascii_pssm='../data/test/pssm/' + out_name + '.pssm',
                                          num_descriptions=10000,
                                          num_alignments=10000,
                                          out='../data/test/alns/' + out_name + '.alns.blast',
                                          num_threads=8)

            cmd()

    for _, sample in tqdm(new_df.iterrows(), desc='Parsing', total=len(new_df)):
        structure = sample['Structure']
        structure = ' ' + structure
        sample_id = sample['ID']
        chain = sample['Chain']
        if not os.path.exists(f'../data/test/pssm/{sample_id}_{chain}.pssm'):
            continue
        with open(f'../data/test/pssm/{sample_id}_{chain}.pssm', 'r') as pssm_file:
            pssm_file.readline()
            pssm_file.readline()
            file_list = []
            offset = False
            position = 0
            for line in pssm_file:
                line = line.rstrip()
                if not line:
                    break
                line = line.split()
                line.append(structure[position])
                position += 1
                if not offset:
                    for i in range(2):
                        line.insert(0, '')
                        offset = True
                file_list.append(line)
            df = pd.DataFrame(file_list)
            df.drop((df.columns[col] for col in range(2, 22)), axis=1, inplace=True)
            df.drop((df.columns[-3:-1]), axis=1, inplace=True)
            df.drop((df.columns[0]), axis=1, inplace=True)
            df.columns = df.iloc[0]
            df = df[1:]
            df.rename(columns={df.columns[0]: "Sequence"}, inplace=True)
            df.rename(columns={df.columns[-1]: "Structure"}, inplace=True)
            df = df[['Structure'] + [col for col in df.columns if col != 'Structure']]
            df.loc[:, 'A':'V'] = df.loc[:, 'A':'V'].astype(float).divide(100)
            df.to_csv(f'../data/test/profile/{sample_id}_{chain}.profile', sep='\t', index=False)
