'''
This script parses the uniref_100.fast file to extract 
the amino acid sequences. The AA seq identifier is used to 
query UniProt in order to retrieve the GO annotations associated with to it.

For each protein in the dataset, the amino acid sequence 
in string format and the list of its go notations are extracted and saved on disk.

This file creates and save proteinsSeqs and proteinsGoes like:
proteinsSeqs=[seq1, seq2]
proteinsGoes=[[go1,go2],[go1]]
'''

from Bio import SeqIO
import pickle
import re
import urllib.request
import time
import yaml

with open('../../hyperparams.yaml', 'r') as f:
    configs=yaml.load(f)
data_dir=configs['data_dir']

url = 'https://www.uniprot.org/uploadlists/'

rawFilename='uniref_100.fasta'
identifiers=[]
sequences=[]

proteinsGoes=[]
batch_size=250
for idx, record in enumerate(SeqIO.parse(data_dir+rawFilename, "fasta")):
    aa_identifier=str(record.id)
    aa_identifier=aa_identifier[aa_identifier.index('_')+1:]
    aa_sequence=str(record.seq)

    identifiers.append(aa_identifier)
    sequences.append(aa_sequence)

print('Number of identifiers extracted:', len(identifiers))
print('Number of sequences extracted:', len(sequences))

for batch_start in range(0, len(identifiers), batch_size):
    time_start=time.time()
    batch_identifiers=identifiers[batch_start: batch_start+batch_size]

    params = {
    'from': 'ACC+ID',
    'to': 'ACC',
    'format': 'tab',
    'query': ' '.join(batch_identifiers),
    'columns': 'go'
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read().decode('utf-8')
        parsed_response=response.split('\n')[1:-1]
        for entry_terms in parsed_response:
            go_terms=re.findall('\[(GO:[0-9]+)\]',entry_terms)
            #start when found [ character. Retrieve 'GO: whatever numbers(+) until ] character found
            proteinsGoes.append(go_terms)

        print('Batch {} in {:.2f}'.format(batch_start//batch_size,time.time()-time_start))

print('\nNumber of sequences GOs:',len(proteinsGoes))
print('Number of sequences extracted:', len(sequences))

with open(data_dir+"proteins_goes", "wb") as fp:
    pickle.dump(proteinsGoes, fp)

with open(data_dir+"proteins_seqs", "wb") as fp:
    pickle.dump(sequences, fp)
