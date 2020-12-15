from contextlib import ExitStack
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
import biotite.application.dssp as dssp
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf
import pandas as pd
from Bio import SeqIO
from sultan.api import Sultan
from tqdm import tqdm
from profile_tools import generate_profiles

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
    s = Sultan()
    pdb_raw_download = pd.read_csv('../data/test/pdb_raw_download.csv', usecols=[0, 1, 2, 3, 4, 5], header=0)
    pdb_raw_download = filter_entries(pdb_raw_download)
    clean_test = reduce_redundancy(pdb_raw_download)
    clean_test[['ID', 'Chain']] = clean_test['ID'].str.split("_", expand=True)

    full_test = pd.DataFrame()
    list_h = ['H']
    list_e = ['E', 'B']
    count = 0
    for _, sample in tqdm(clean_test.iterrows(), total=len(clean_test), desc='Generating DSSP'):
        sample_id = sample['ID']
        chain = sample['Chain']
        sequence = sample['Sequence']
        file_name = rcsb.fetch(sample_id, "mmtf", gettempdir())
        mmtf_file = mmtf.MMTFFile.read(file_name)
        array = mmtf.get_structure(mmtf_file, model=1)
        structure = array[struc.filter_amino_acids(array)]
        structure = structure[structure.chain_id == chain]
        structure.set_annotation('chain_id', ['A' for el in structure.chain_id])
        sse = dssp.DsspApp.annotate_sse(structure)
        sse = ['H' if ss in list_h else 'E' if ss in list_e else 'C' for ss in sse]
        sse = ''.join(sse)
        if len(sse) != len(sequence):
            tqdm.write(f'Length mesmatch between sequence and structure for {sample_id}_{chain}')
            continue
        sample['Structure'] = sse
        full_test = full_test.append(sample)
        count += 1
        if count == 10:
            break
    full_test.set_index(['ID', 'Chain'], inplace=True)
    generate_profiles(full_test, '../data/test/')


