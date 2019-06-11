import yaml

data={
    'labels': 'hello'
}


with open('data.yaml', 'w') as outfile:
    yaml.dump(data, outfile)

with open("data.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

with open('data.yaml', 'a') as outfile:
    yaml.dump({'categories':'hello'}, outfile)

with open("data.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded)