import os
import pandas as pd
from sultan.api import Sultan
from Bio import SeqIO


def filter_entries(pdb_download):
    pdb_download = pdb_download[pdb_download['Sequence'].str.contains('X') == False]
    pdb_download['Chain Length'] = pd.to_numeric(pdb_download['Chain Length'], errors='coerce')
    pdb_download = pdb_download[pdb_download['Chain Length'].isna() == False]
    pdb_download = pdb_download.reset_index(drop=True)
    return pdb_download


def reduce_redundancy(pdb_download):
    out_path = 'tmp/unclustered.fasta'

    def prepare_fasta():
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(out_path, 'a') as test_unclustered:
            for entry in pdb_download.index:
                sequence = pdb_download.loc[entry, 'Sequence']
                pdb_id = pdb_download.loc[entry, 'PDB ID']
                chain = pdb_download.loc[entry, 'Chain ID'].split(',')[0]
                print('>', pdb_id, '_', chain, '\n', sequence, sep='', end='\n', file=test_unclustered)

    def reduce_internal():
        with Sultan.load(sudo=False) as s:
            os.chdir('tmp')
            unclustered_path = 'unclustered.fasta '
            db_path = '../db/testDB '
            clstr_path = 'test.clstr '
            tmp_path = '. '
            representatives_path = 'representatives '
            representatives_fasta_path = 'representatives.fasta '
            s.cat('unclustered.fasta').pipe().\
                mmseqs('createdb stdin sequenceDB').run()
            s.mmseqs('cluster sequenceDB clstr . --min-seq-id 0.3 -c 0.5 --cov-mode 0').run()
            s.mmseqs('createsubdb clstr sequenceDB representatives').run()
            s.mmseqs('convert2fasta representatives representatives.fasta').run()

    def reduce_external():
        def blast_jpred():
            with Sultan.load(sudo=False) as s:
                s.blastp('-query representatives.fasta '
                         '-out representatives_similar.tsv '
                         '-db ../db/jpred.fasta '
                         '-outfmt 6 '
                         '-evalue 0.01').run()
                # to_keep = ['representatives.fasta']
                s.rm('-r *')

        def purge_redundant():
            representatives_similar = pd.read_csv('representatives_similar.tsv',
                                                  sep='\t',
                                                  names=['Query',
                                                         'Subject',
                                                         'Identity',
                                                         'Length',
                                                         'mismatches',
                                                         'gap opens',
                                                         'Query start',
                                                         'Quesry end',
                                                         'Sequence start',
                                                         'Sequence end',
                                                         'E-value',
                                                         'Bit score']
                                                  )

            representatives = list()
            for record in SeqIO.parse("representatives.fasta", "fasta"):
                representatives.append([record.id, str(record.seq)])

            representatives = pd.DataFrame(representatives, columns=['ID', 'Sequence'])
            similar_id = representatives_similar[representatives_similar['Identity'] > 30]
            similar_id = similar_id['Query'].values.tolist()
            return representatives
        blast_jpred()
        representatives = purge_redundant()
        return representatives
    prepare_fasta()
    reduce_internal()
    representatives = reduce_external()
    return representatives


os.chdir('./../data/test/')
pdb_raw_download = pd.read_csv('pdb_raw_download.csv', usecols=[0, 1, 2, 3, 4, 5])
pdb_raw_download = filter_entries(pdb_raw_download)
full_test = reduce_redundancy(pdb_raw_download)
full_test.to_csv('full_test.tsv', sep='\t')
