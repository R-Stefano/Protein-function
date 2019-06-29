import pandas as pd
import re 

'''
This script extracts the reference of the most similar sequence based on BLAST search.
The reference of each test sequence is saved on a file in order to then 
query Uniprot to get the GO notations
'''

data = pd.read_csv("results_queries.csv", header=None)

data.columns = ['sequence', 'reference', 'identity', '', '', '', '', '','','','e_value','bit_score']
print(data.head(20))

test_examples=142688
blast_ref_predictions_file=open('blast_ref_predictions.txt', 'w+')

for i in range(test_examples):
    print('Parsing result', i)
    seq_query='seq'+str(i) #seq0, seq1 ...
    seq_res=data.loc[data['sequence'] == seq_query]
    try:
        first_res=seq_res.iloc[0]
        if (int(first_res['identity'])==100):
            #exclude first result, otherwise blast results will be 100% accurate
            second_res=seq_res.iloc[1]
            seq_ref_res=second_res['reference']        
        else:
            seq_ref_res=first_res['reference']
        
        #retrieve only reference
        #sp|Q5MZP5|COAX_SYNP6 -> Q5MZP5
        start_crop=seq_ref_res[seq_ref_res.index('|')+1:]
        ref=start_crop[:start_crop.index('|')]
    except:
        ref='None'


    blast_ref_predictions_file.write('{}\n'.format(ref))


blast_ref_predictions_file.close()