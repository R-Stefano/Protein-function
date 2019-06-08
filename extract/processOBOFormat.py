import obonet
import networkx
import pickle
'''
I have found this library called obonet which allows to quickly parse the .obo file 
and creating a graph based on the relationships described in the file. 

library tutorial
https://github.com/dhimmel/obonet/blob/master/examples/go-obonet.ipynb

Let's start by predicting the macro-class which is only 3 classes compared to the 40k.

'''
graph = obonet.read_obo('go.obo')

#Number of nodes
print(len(graph))
#Number of edges
print(graph.number_of_edges())

labelsData={}
unique_classes={}
for n in graph.nodes():
    nodeData=graph.node[n]
    unique_classes[nodeData['namespace']]=1
    labelsData[n]={
        'class': nodeData['namespace'],
        'type': nodeData['def']
    }
print('Number classes:', len(unique_classes))
print('Number of unique GOs:', len(labelsData))

dict_unique_goes={
    'dict_goes':labelsData,
    'classes':list(unique_classes.keys())
}
with open("dict_unique_goes", "wb") as fp:
    pickle.dump(dict_unique_goes, fp)

'''
{
    'name': 'NAD-dependent histone deacetylase activity (H3-K18 specific)', 
    'namespace': 'molecular_function', 
    'def': '"Catalysis of the reaction: histone H3 N6-acetyl-L-lysine (position 18) + H2O = histone H3 L-lysine (position 18) + acetate. This reaction requires the presence of NAD, and represents the removal of an acetyl group from lysine at position 18 of the histone H3 protein." [EC:3.5.1.17, GOC:sp, PMID:22722849, RHEA:24548]', 
    'is_a': ['GO:0017136', 'GO:0034739'], 
    'created_by': 'paola', 
    'creation_date': '2012-09-05T13:16:39Z'}

'''