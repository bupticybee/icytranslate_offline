# icytranslate_offline
This project is the offline part of [Icytranslate](http://translate.icybee.cn/) , an English-Chinese translate platform. The output of this project is a translate model, which is the core component of icytranslate.

# data preparation
We use UM-corpus as our default training dataset, which can be applied here:

[http://nlp2ct.cis.umac.mo/um-corpus/](http://nlp2ct.cis.umac.mo/um-corpus/)

# User your own dataset
Althrough UM-corpus is a fine dataset, we encourage you to use your own dataset and report your results. If you want to use other datasets , you might need to modify the code in segment.py and change the train_prefix and test_prefix to the actual data dir.

Beweare the dataset you use should have the same data structure as the UM-crop otherwise you might want to read the ```segment.py``` and modify some of the code in it.

# data preprocessing

### tokenlizer
We first need to process the corpus into series of words, run ```python segment.py``` to do that. The output should be ```segment_train.pkl``` in ```middleresult``` dir.

We process all english sentences in to words in lower case , and process all chinese sentences into lists of chinese characters.

### encode the tokenlized series
The next step is to convert the tokenlizered sentences into sequences of words, doing that, you only need to run

```python word2index.py --max_words=[max words in a sentence that you want]```

# model training
Now we can train our model. You may find a ```align-and-translate-char``` ipynb file in the folder, open the file with an IDE or jupyter notebook, and follow the steps there, you will get the model trained and a test bleu around 0.22.

# dependences
```
tensorflow 1.2.0 for neural network
jieba for english word tokenlizer
nltk to calculate bleu score
sklearn , numpy as toolkit
```
