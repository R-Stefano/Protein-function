'''
This script loads the file containing the protein reference for each 
test example, query Uniprot to get the GO notations and save the Go notations 
on a file for later comparison with the deep learning model.
'''
from bioservices import UniProt
import re
import pickle

service = UniProt()
blast_ref_predictions=open('blast_ref_predictions.txt', 'r')
batch_size=25000

predictions=[]
for idx, line in enumerate(blast_ref_predictions):
    ref=line.rstrip()#remove newline indentation
    if ref=="None":
        goes_list=[]
    else:
        print('\n\nObtaining GOs for', ref)
        res=service.search(ref, frmt="tab", columns="go")
        #print('Result:\n'+res)
        
        goes_list=[]
        #parse the result to extract the GO notations
        bracket_idx=0
        while bracket_idx>-1:
            bracket_idx=res.find('[')
            res=res[bracket_idx+1:]
            go=res[:res.find(']')]
            goes_list.append(go)

        #remove last result which is a duplicate of the second-last result
        goes_list=goes_list[:-1]

        print('Number GOs found', len(goes_list))

    predictions.append(goes_list)

    if (idx%batch_size==0 and idx!=0):
        print('saving')
        pickle.dump(predictions, open( "blast_predictions/blast_goes_predicted_"+str(idx//batch_size), "wb" ))
        predictions=[]


pickle.dump(predictions, open( "blast_predictions/blast_goes_predicted_last", "wb" ))