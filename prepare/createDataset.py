import numpy as np
import pickle
import sys
import tfapiConverter as tfconv

'''
There are 560118 sequences that must be hot encoded (for the moment).
So, first to count how many unique amino acids there are. 
Then, create the hot encode vectors
Finally, for each amino acid associate its own hot vector.
Store the results in a new file.

I had a memory problems when I try to dsave on disk the batches.
SO, I thought thta maybe isterad of reinventing the wheel, it could worth a try to 
explore if tensorflow cameup with some API. 

The data are encoded as uint8 (0,255) which is the most space efficient format.
For example, a matrix of [597, 512, 25)] in flat64=61.1 MB
                                            in uint8=7.6 MB
This is possible because The vectors only have values 0 and 1

This class takes as input the pickles about:
sequence as string, goes for each sequence, category of the goes

output:
tfrecords ready to be load and processed
'''

'''
the for loop is a lstm. The while must explicitly implement when get out of the loop, the for loop can keep going.

No, maybe the while is better, it process internally, never output anything, are the others
that read the internal state of the while. So, they must be processed in PARALLEL.

it's like that each while loop is in charge of a thoughts-reading process. The loops try to 
read other loops. 

When they do it could be at each timestep or they output when to read. So, there is an 
internal state which decides when reading if it exceed a given threshold.
'''

#Keep the sequences with a length max of:
max_length_amino=512
file_batch_size=20000

#save processed data on disk
def prepareDataset(inputData, labelData, filename):
    print('Saving:', filename)
    data={
        'X': inputData,
        'Y': labelData
    }

    tfconv.generateTFRecord(inputData, labelData, 'data/'+filename+'.tfrecords')


def applyMask(dirtyData, dirty_idxs):
    if (type(dirtyData)!=type(np.asarray([]))):
        dirtyData=np.asarray(dirtyData)

    returnData= dirtyData[np.logical_not(dirty_idxs)]

    return returnData

def preprocessLabels(goes_seqs, dict_goes, classes):
    #encode the categories
    hot_vecs=np.identity(len(classes))

    #get GO's category and retrieve category hot encode
    hot_cats_seqs=np.zeros((len(goes_seqs),len(classes)))
    mask=[]
    for i, goes_list in enumerate(goes_seqs):
        hot_cat_seq=[]
        for goes in goes_list:
            try:
                className=dict_goes[goes]['class']
                idx=classes.index(className)
                hot_category=hot_vecs[idx]
                hot_cat_seq.append(hot_category)
            except:
                #example goes not identified, discard it
                continue

        if hot_cat_seq==[]:
            mask.append(True)
        else:
            mask.append(False)
        
        #create a single label from labels.
        #es. if ex1 has labels 0, 2 but not 1
        #ex1=[[1,0,0], [0,0,1]] but not [0,1,0]
        #its single label is ex1=[1,0,1]
        hot_cat_seq=np.sum(hot_cat_seq, axis=0)

        #keep only 1 or 0
        hot_cat_seq=(hot_cat_seq >0)

        hot_cats_seqs[i]=hot_cat_seq

    #return [seqs, hot_vec]
    return np.array(hot_cats_seqs,dtype=np.uint8), mask

def preprocessInpuData(seqs_str, aminos_list, max_length_amino): 
    '''
    input:
        seqs_str(list): each element is a string of amino acids
        aminos_list(list): each element is a single amino acid
        max_length_amino(int): discard the sequences longer than this number 
    ''' 
    #create hot encode for each unique amino
    hot_aminos=np.identity(len(aminos_list))

    #prepare the padded batch
    hot_aminos_seqs=[]
    mask=[]
    for i, seq_str in enumerate(seqs_str):
        hot_amino_seq=np.zeros((max_length_amino, len(aminos_list)))
        if len(seq_str)<max_length_amino:
            enumerator=enumerate(seq_str)
            mask.append(False)
        else:
            enumerator=enumerate(seq_str[:max_length_amino])
            mask.append(True)

        for j, amino in enumerator:
            idx=aminos_list.index(amino)
            hot_amino_seq[j]=hot_aminos[idx]

        hot_aminos_seqs.append(hot_amino_seq)

    #assign hot_amino to each seq's amino
    return np.array(hot_aminos_seqs, dtype=np.uint8), mask

def main():
    print('Importing files..')
    #FOR LABEL DATA:
    #get the GOs for each example
    with open("../extract/seqs_goes_str", "rb") as fp:
        goes_seqs = pickle.load(fp)

    #get the dictionary of unique GOs
    with open("../extract/dict_unique_goes", "rb") as fp:
        labelData=pickle.load(fp)
        dict_goes=labelData['dict_goes']
        classes=labelData['classes']

    #FOR INPUT DATA:
    #get the sequences as string
    with open("../extract/seqs_str", "rb") as fp:
        inputData = pickle.load(fp)
        seqs_str=inputData['seqs']
        aminos_list=inputData['aminos']

    tot_examples=0
    print('Creating chunks dataset..')
    for startBatch in range(1000, len(seqs_str), file_batch_size):
        endBatch=startBatch+file_batch_size

        batch_goes_seqs=goes_seqs[startBatch:endBatch]
        batch_seqs_str=seqs_str[startBatch:endBatch]

        #Labels: return [seqs, hot_vec]
        dirty_labelData, mask_empty_examples=preprocessLabels(batch_goes_seqs, dict_goes, classes)

        #Inputs: return [seqs, num_aminos, hot_vec]
        dirty_inputData, mask_too_long_examples=preprocessInpuData(batch_seqs_str, aminos_list, max_length_amino)

        #merge the two masks (or operator)
        dirty_idxs=np.logical_or(mask_empty_examples,mask_too_long_examples)

        #remove dirty examples
        batch_inputData=applyMask(dirty_inputData, dirty_idxs)
        batch_labelData=applyMask(dirty_labelData, dirty_idxs)
        print('Ready input data:', batch_inputData.shape, 'values type', batch_inputData.dtype, 'size:', sys.getsizeof(batch_inputData)*1e-6,'MB')
        print('Ready label data:', batch_labelData.shape, 'values type', batch_labelData.dtype, 'size:', sys.getsizeof(batch_labelData)*1e-6,'MB')

        tot_examples+=batch_inputData.shape[0]

        filename='dataset_'+str(startBatch//file_batch_size)
        prepareDataset(batch_inputData, batch_labelData, filename)
        print('\n')
    print('Total number of examples', tot_examples)

if __name__ == '__main__':
    main()