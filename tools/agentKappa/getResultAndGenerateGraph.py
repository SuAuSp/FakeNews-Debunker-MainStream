import pandas as pd
import numpy as np
import json

with open('config.json', 'r') as f:
    config = json.load(f)

modelConfig = config['modelConfig']
datasetConfig = config['datasetConfig']
promptConfig = config['promptConfig']

taskName = datasetConfig['taskName']
file = datasetConfig['file']
text = datasetConfig['textLabel']
values = datasetConfig['valueLabels']
width = datasetConfig['widthValue']
sampleNum = datasetConfig['testNum']
promptNum = datasetConfig['promptNum']
timesLimit = datasetConfig['timesLimit']
binwidths = datasetConfig['binwidths']

with open(f'{taskName}KappaResult.json', 'r') as f:
    kappaResults = json.load(f)

results = {"valueType": [], "testValue": [], "kappa": []}

for value in values:
    for test in kappaResults[value]:
        results['valueType'].append(value)
        results['testValue'].append(test['testValue'])
        results['kappa'].append(test['kappa'])

resultGraph = pd.DataFrame(results)
resultGraph.to_csv(f'{taskName}KappaResultGraph.csv')

