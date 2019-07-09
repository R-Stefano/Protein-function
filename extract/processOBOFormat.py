import obonet
import pickle

'''
This script parses the go.obo file to extract 
the go notations and their function description.

It creates and save a go_dictionary like:

go_dictionary:
{
    'GO:0017136': 'NAD-dependent histone deacetylase activity (H3-K18 specific)'
}
'''
graph = obonet.read_obo('go.obo')

#Number of nodes
print(len(graph))
#Number of edges
print(graph.number_of_edges())

go_dictionary={}
print('Mapping go notation to protein function..')
for go in graph.nodes():
    nodeData=graph.node[go]
    go_dictionary[go]=nodeData['name']

print('Number classes:', len(go_dictionary.keys()))

with open("go_dictionary", "wb") as fp:
    pickle.dump(go_dictionary, fp)

'''
nodeData:
'GO:0000001':{
    'name': 'NAD-dependent histone deacetylase activity (H3-K18 specific)', 
    'namespace': 'molecular_function', 
    'def': '"Catalysis of the reaction: histone H3 N6-acetyl-L-lysine (position 18) + H2O = histone H3 L-lysine (position 18) + acetate. This reaction requires the presence of NAD, and represents the removal of an acetyl group from lysine at position 18 of the histone H3 protein." [EC:3.5.1.17, GOC:sp, PMID:22722849, RHEA:24548]', 
    'is_a': ['GO:0017136', 'GO:0034739'], 
    'created_by': 'paola', 
    'creation_date': '2012-09-05T13:16:39Z'}

'''