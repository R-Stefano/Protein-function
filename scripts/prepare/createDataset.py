'''
This script loads the raw data, clean them and prepare for training. 

It convertes each amino acid sequence into a list of idexes. 
The indexes indicate the amino acid. such as [ARNDCA] -> [0,1,2,3,4,0] 

Same, it converts the go notations into a list of indexes. 
[GO1, GO2, GO3] -> [0,1,2] 

Then, it discards the examples that don't meet the criteria (max sequence length, Go notation)

Finally, it creates train and test datasets in tfrecord format.
'''
import numpy as np
import pickle
import sys
import yaml
from sklearn.model_selection import train_test_split


file_batch_size=500000
with open('../../hyperparams.yaml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']
train_data_dir=configs['train_data_dir']
test_data_dir=configs['test_data_dir']
shared_scripts_dir=configs['shared_scripts']
max_seq_length=configs['max_length_aminos']
min_seq_length=configs['min_length_aminos']
seed=configs['seed']
test_size=configs['test_size']
sys.path.append(shared_scripts_dir)
import tfapiConverter as tfconv

proteins_goes=pickle.load(open(data_dir+"proteins_goes", "rb"))
proteins_seqs=pickle.load(open(data_dir+"proteins_seqs", "rb"))
dataset_config=yaml.load(open(data_dir+'dataset_config.yaml', 'r'))

label_gos=dataset_config['available_gos']
mapped_go_subterms=dataset_config['mapped_gos']
unique_aminos=dataset_config['unique_aminos']

#save processed data on disk
def prepareDataset(inputData, labelData, filename):
    print('Saving', filename)

    X_train, X_test, y_train, y_test = train_test_split(inputData, labelData, test_size=test_size, random_state=seed)

    #tfconv.generateTFRecord(X_train, y_train, train_data_dir+filename+'.tfrecords')
    #tfconv.generateTFRecord(X_test, y_test, test_data_dir+filename+'.tfrecords')
    np.save(train_data_dir+'input_'+filename+'.npy', np.stack(X_train))
    np.save(train_data_dir+'label_'+filename+'.npy', y_train)

    np.save(test_data_dir+'input_'+filename+'.npy', np.stack(X_test))
    np.save(test_data_dir+'label_'+filename+'.npy', y_test)


def applyMask(dirtyData, dirty_idxs):
    if (type(dirtyData)!=type(np.asarray([]))):
        dirtyData=np.asarray(dirtyData)

    returnData= dirtyData[np.logical_not(dirty_idxs)]

    return returnData

def preprocessLabels(goes_seqs, unique_goes, mapped_goes):
    '''
    This function index the go notations for each sequence

    Args:
        goes_seques (list): each element is a list of go notations for a protein
        unique_goes (list): list of unique goes
        mapped_goes (dictionary): a dictionary where the key is a go subterm and the value is a list of 
                                  labels. The labels are the parents GOs of the key GO
    
    Return:
        hot_cats_seqs (list): each element is a list of indexed go notations
        mask (list): list of bools, if True the example must be discarded
    '''
    #get GO's category and retrieve category hot encode
    hot_cats_seqs=[]
    mask=[]
    for i, goes_list in enumerate(goes_seqs):
        hot_cat_seq=[]
        for go in goes_list:
            if go in mapped_goes:
                #get all the parents terms
                parent_terms=mapped_goes[go]
                for p_go in parent_terms:
                    label_idx=unique_goes.index(p_go)
                    if label_idx not in hot_cat_seq:
                        hot_cat_seq.append(label_idx)

        if hot_cat_seq==[]:
            mask.append(True)
        else:
            mask.append(False)

        hot_cats_seqs.append(hot_cat_seq)

    #return [goes_idxs_lists], [bools]
    return hot_cats_seqs, mask

def preprocessInpuData(seqs_str, unique_aminos): 
    '''
    This function index the amino acids for each sequence
    Args:
        seqs_str(list): each element is a string of amino acids
        unique_aminos(list): list of unique ami42no acids

    Return:
        hot_aminos_seqs (list): each element is a list of indexed amino acids
        mask (list): list of bools, if True the example must be discarded
    ''' 

    hot_aminos_seqs=[]
    mask=[]
    for i, seq_str in enumerate(seqs_str):
        #discard sequences shorter and longer than given thresholds
        l_seq=len(seq_str)
        if ((l_seq>=min_seq_length) and (l_seq<=max_seq_length)):
            encoded_seq=np.zeros((max_seq_length), dtype=np.int8)-1
            for seq_i, aa in enumerate(seq_str):
                encoded_seq[seq_i]=unique_aminos.index(aa)
            mask.append(False)
        else:
            encoded_seq=[]
            mask.append(True)

        hot_aminos_seqs.append(encoded_seq)

    #assign hot_amino to each seq's amino
    return hot_aminos_seqs, mask


tot_examples=0
print('Creating dataset..')
for startBatch in range(0, len(proteins_goes), file_batch_size):
    endBatch=startBatch+file_batch_size

    batch_proteins_goes=proteins_goes[startBatch:endBatch]
    batch_proteins_seqs=proteins_seqs[startBatch:endBatch]

    #Labels: return [seqs, hot_vec]
    print('Preprocessing labels..')
    dirty_labelData, mask_empty_examples=preprocessLabels(batch_proteins_goes, label_gos, mapped_go_subterms)
    #Inputs: return [seqs, num_aminos, hot_vec]
    print('Preprocessing input data..')
    dirty_inputData, mask_too_long_examples=preprocessInpuData(batch_proteins_seqs, unique_aminos)

    #merge the two masks (or operator)
    dirty_idxs=np.logical_or(mask_empty_examples,mask_too_long_examples)

    #remove dirty examples
    print('Removing dirty examples..')
    batch_inputData=applyMask(dirty_inputData, dirty_idxs)
    batch_labelData=applyMask(dirty_labelData, dirty_idxs)
    print('Ready input data:', batch_inputData.shape, 'values type', batch_inputData.dtype, 'size:', sys.getsizeof(batch_inputData)*1e-4,'MB')
    print('Ready label data:', batch_labelData.shape, 'values type', batch_labelData.dtype, 'size:', sys.getsizeof(batch_labelData)*1e-4,'MB')

    tot_examples+=batch_inputData.shape[0]

    filename='dataset_'+str(startBatch//file_batch_size)
    prepareDataset(batch_inputData, batch_labelData, filename)

print('Total number of examples', tot_examples)