import pandas as pd
from profile_tools import generate_profiles
from sultan.api import Sultan

if __name__ == '__main__':
    summary = pd.read_csv('../data/training/jpred4_summary.tsv', sep='\t', header=0, index_col=0)
    jpred = pd.DataFrame(columns=['Sequence', 'Structure'])
    s = Sultan()
    s.unzip('-o ../data/training/jpred_dssp.zip -d /tmp').run()
    s.unzip('-o ../data/training/jpred_fasta.zip -d /tmp').run()
    print('Parsing fasta and dssp files')
    for _, sample in summary.iterrows():
        jpred_id = sample.name
        sample.drop(sample.index, inplace=True)
        with open(f'/tmp/{jpred_id}.fasta') as fasta_file, \
                open(f'/tmp/{jpred_id}.dssp') as dssp_file:
            fasta_file.readline(), dssp_file.readline()
            sequence = fasta_file.readline().rstrip()
            structure = dssp_file.readline().rstrip()
            structure = structure.replace('-', 'C')
            sample['Sequence'] = sequence
            sample['Structure'] = structure
            jpred = jpred.append(sample)
    generate_profiles(jpred, '../data/training/')
