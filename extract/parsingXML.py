'''
This script parses the uniprot.xml file to extract 
the amino acid sequences their go notations.

For each protein in the dataset, the amino acid sequence 
in string format and the list of its go notations are extracted and saved on disk.

This file creates and save proteinsSeqs and proteinsGoes like:
proteinsSeqs=[seq1, seq2]
proteinsGoes=[[go1,go2],[go1]]
'''
import xml.etree.ElementTree as ET
import pickle
import numpy as np

rawFilename='uniprot-filtered-reviewed_yes.xml'

print('Parsing the xml to extract the data..')
context = ET.iterparse(rawFilename, events=('start', 'end'))

proteinsSeqs=[]
proteinsGoes=[]

count=0
bufferGOList=[] #stores the goes for each protein
for evt, element in context:
    if count%1000000==0 and count>1000000:
        print('Number examples retrieved:',len(proteinsSeqs))


    if (('entry' in element.tag) and (evt=='end')):
        proteinsGoes.append(bufferGOList)
        bufferGOList=[]

        #throw away the element
        element.clear()
    
    #Retrieve the GO ids
    if (('dbReference' in element.tag) and (element.get('type')=="GO") and (evt=='end')):
        bufferGOList.append(element.get('id'))

    #Retrieve the amino acid sequence and remove new line character
    try:
        if (('sequence' in element.tag) and (evt=='end')):
            seq=element.text.replace('\n','')

            proteinsSeqs.append(seq)
    except:
        continue

    count +=1

print('Number of prot sequences:',len(proteinsSeqs))
print('Number of prot goes:',len(proteinsGoes))

with open("proteins_seqs", "wb") as fp:
    pickle.dump(proteinsSeqs, fp)

with open("proteins_goes", "wb") as fp:
    pickle.dump(proteinsGoes, fp)
