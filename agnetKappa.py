import ollama
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import random

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

# Dataframe Content must be a Value for classification and send to LLM for Labeling.
# Need more sample for labeling to except LLM output error

dataFrame = pd.read_csv(file)
dataFrame.drop_duplicates(subset=[text], keep='first', inplace=True)

LLMresults = {}
for value in values:
        
    dataTexts, dataValues = dataFrame[text].tolist(), dataFrame[value].tolist()

    taskGuidelines = promptConfig['taskGuidelines']

    # Set Value
    # TwoType Classification
    valueRangeMax = np.max(dataValues)
    valueRangeMin = np.min(dataValues)

    # Set a Range for Every Type
    testRange = binwidths

    LLMresult = []
    # Begin a While until it Finished
    for testValue in tqdm(testRange):
        # Generate Labels, Higher than the testValue is high, Lower is same
        dataFrame['label'] = dataFrame[value].map(lambda x: x > testValue)

        # Then get the Sample for Test.
        # Positive Sample and Negative Sample
        positiveSample = dataFrame[dataFrame['label'] == True][text].sample(sampleNum*2).tolist()
        negativeSample = dataFrame[dataFrame['label'] == False][text].sample(sampleNum*2).tolist()

        subDataFrame = dataFrame[dataFrame['text'].isin(positiveSample + negativeSample) == False]

        # Get some Prompt Sample for few-shot
        positivePrompt = subDataFrame[subDataFrame['label'] == True][text].sample(promptNum).tolist()
        negativePrompt = subDataFrame[subDataFrame['label'] == False][text].sample(promptNum).tolist()

        # Then Generate the Prompt for LLM
        
        # Pack Sample to be a Example
        positivePrompt = '\n'.join([ f"Example: {sample}\nOutput:1" for sample in positivePrompt])
        negativePrompt = '\n'.join([ f"Example: {sample}\nOutput:0" for sample in negativePrompt])

        # Get testExamples
        testExamples = [f"Input: {sample}\nOutput:" for sample in positiveSample + negativeSample]
        random.shuffle(testExamples) # 打乱一次testExamples
        retList = []

        for index,testExample in enumerate(testExamples):
            prompt = [
                taskGuidelines,
                'Some examples with their output answers are provided below:',
                positivePrompt,
                negativePrompt,
                'Now I want you to label the following example,Provide a “0” or “1” answer only. Do not explain or add anything else:',
                testExample,
            ]
            prompt = '\n'.join(prompt)
            retFirst, retSecond = None, None

            timesAcc = 0
            while ( retFirst not in ['0', '1'] or retSecond not in ['0', '1'] ) and timesAcc < timesLimit:
                timesAcc += 1
                # Maybe Need More Model to Labeling
                # First Agent
                response = ollama.generate(
                    model='llama3', 
                    prompt=prompt,
                    stream=False,
                    options=modelConfig,
                )
                retFirst = response['response']
        
                # Second Agent
                response = ollama.generate(
                    model='gemma2', 
                    prompt=prompt,
                    stream=False,
                    options=modelConfig,
                )
                retSecond = response['response']

                print(f"Log: LLMLabeling Result {index}, First:{retFirst}, Second:{retSecond}")
            
            if timesAcc == timesLimit:
                continue
            else:
                # Then ret must be 0 or 1
                retFirst, retSecond = int(retFirst), int(retSecond)
                retList.append([retFirst, retSecond])
            
            # break for if achieve the num
            if len(retList) == sampleNum*2:
                break
            
        retList = np.transpose(retList).tolist()
        LLMresult.append({
            'testValue' : testValue,
            'testLabel' : retList,
        })

    # Out of Range, Then Calculate Kappa

    for index, sample in enumerate(LLMresult):
        testValue = sample['testValue']
        textLabel = sample['testLabel']
        retFirst = textLabel[0]
        retSecond = textLabel[1]
        kappaResult = cohen_kappa_score(retFirst, retSecond)
        LLMresult[index]['kappa'] = kappaResult

    LLMresults[value] = LLMresult

with open(f'{taskName}KappaResult.json', 'w') as f:
    json.dump(LLMresults, f, indent=4)


