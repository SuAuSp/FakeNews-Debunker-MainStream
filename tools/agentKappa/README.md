# AgentKappa
修改`config.json`来更改

参数:
```python
"datasetConfig":{
        "taskName" : "Your task name", # input your task name
        "file" : "Your dataset(must be csv)", # input your dataset
        "valueLabels" : ["your label value"], # csv label(columns)
        "textLabel" : "your text label",  # csv column
        "widthValue" : 0.05, # 暂时没有用处
        "testNum" : 30, # send to Agent to Labeling Num, 30 means 60 sample to label 
        "promptNum" : 5,  # sent to Agent for labeling, 5 means 10 example
        "timesLimit" : 10,  # LLM result may output error, it will try 10 times,or continue to next until reach testNum
        "binwidths" : [0.2,0.3,0.4,0.5,0.6,0.7,0.8] # the threshold need to test
    },
"promptConfig":{
    "taskGuidelines" : "You are an expert at analyzing the toxicity of tweets. Your job is to classify the provided texts into one of the following labels: [0, 1], 0 means this text has low toxicity and 1 is high.Provide a “0” or “1” answer only. Do not explain or add anything else." # the prompt you need to change in your task
}
```
