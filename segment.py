import jieba
from wordsegment import segment
import os
train_prefix = 'data/UM-Corpus/data/Bilingual/'
test_prefix = 'data/UM-Corpus/data/Testing/'
train_classes = os.listdir(train_prefix)

train_files = [os.listdir(train_prefix + i)[0] for i in train_classes]

train_chinese = []
train_english = []
for i,j in zip(train_classes,train_files):
    filename = os.path.join(train_prefix,i,j)
    with open(filename,encoding='utf-8') as fhdl:
        flag = 0
        for line in fhdl:
            if flag == 0:
                train_english.append(line.strip())
            else:
                train_chinese.append(line.strip())
            flag = flag ^ 1
      
print(len(train_chinese),len(train_english))
from utils import *
pb = ProgressBar(worksum=len(train_chinese),auto_display=False)

pb.startjob()
train_token_chinese = []
train_token_english = []
num = 0
with open('middleresult/segmented_train.txt','w',encoding='utf-8') as whdl:
    for ch,en in zip(train_chinese,train_english):
        num += 1
        token_en = [i.lower() for i in jieba.cut(en) if i.strip()]
        token_ch = [i for i in ch if i.strip()]
        train_token_chinese.append(token_ch)
        train_token_english.append(token_en)
        whdl.write("{}\n".format(' '.join(token_en)))
        whdl.write("{}\n".format(' '.join(token_ch)))
        pb.complete(1)
        if num % 32 == 0:
            pb.display_progress_bar()
            
import pickle
with open('middleresult/segmented_train.pkl','wb') as whdl:
    pickle.dump((train_token_chinese,train_token_english),whdl)