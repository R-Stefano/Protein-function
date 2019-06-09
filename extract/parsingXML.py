import xml.etree.ElementTree as ET
import pickle
import numpy as np

rawFilename='uniprot-filtered-reviewed_yes.xml'

print('Parsing the xml to extract the data..')
context = ET.iterparse(rawFilename, events=('start', 'end'))

inputSeqs=[]
unique_input_amino={}
length_seqs=[]
labelData=[]
count=0

bufferGOList=[]
for evt, element in context:
    if count%1000000==0 and count>1000000:
        print('Number examples retrieved:',len(inputSeqs))
        print('longest sequence:', max(length_seqs))
        print('average length:',np.mean(length_seqs))
        print('var length:',np.var(length_seqs))        
        print('\n\n\n')


    if (('entry' in element.tag) and (evt=='end')):
        labelData.append(bufferGOList)
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

            length_seqs.append(len(seq))
            #See how many unique amino-acids there are
            for a in seq:
                unique_input_amino[a]=1

            inputSeqs.append(seq)
    except:
        continue

    count +=1

print('length:',len(inputSeqs))
print('length:',len(labelData))
print('Number unique amino-acids:', len(unique_input_amino))
print('List unique amino-acids:', list(unique_input_amino.keys()))
'''
List unique amino-acids: 
['M', 'F', 'K', 'V', 'E', 'N', 'A', 'P', 'I', 'L', 'W', 'D', 'S', 'Q', 'R', 
'G', 'C', 'T', 'Y', 'H', 'U', 'X', 'B', 'Z', 'O']

'''

inputData={
    'seqs':inputSeqs, 
    'aminos':list(unique_input_amino.keys())
    }
with open("seqs_str", "wb") as fp:
    pickle.dump(inputData, fp)

with open("seqs_goes_str", "wb") as fp:
    pickle.dump(labelData, fp)
