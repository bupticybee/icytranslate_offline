import pickle
from utils import ProgressBar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_words", type=int,
                    help="max words of the sentences tokened",default=-1)
parser.add_argument("--test_lines", type=int,
                    help="line number used to test the program,leave empty to process full text",default=-1)
parser.add_argument("--vac_dict_en", type=int,
                    help="dictionary size for english",default=60000)
parser.add_argument("--vac_dict_ch", type=int,
                    help="dictionary size for chinese",default=60000)
args = parser.parse_args()


print('reading pkl data...')
with open('middleresult/segmented_train.pkl','rb') as fhdl:
    (train_token_chinese,train_token_english) = pickle.load(fhdl)
if args.test_lines != -1:
    train_token_chinese = train_token_chinese[:args.test_lines]
    train_token_english = train_token_english[:args.test_lines]
print(len(train_token_chinese),len(train_token_english))

import random
index = random.randint(0,len(train_token_chinese))
print(train_token_chinese[index])
print(train_token_english[index])

from collections import Counter
from functools import reduce
print('counting...')

ch_dic = {}
def get_most_common(a1,a2):
    temp_dict1 = {}
    temp_dict2 = {}
    pb = ProgressBar(worksum=len(a1),auto_display=False)
    pb.startjob()
    num = 0
    for s1,s2 in zip(a1,a2):
        num += 1
        pb.complete(1)
        if args.max_words != -1 and len(s1) > args.max_words:
            continue
        for w1 in s1:
            temp_dict1.setdefault(w1,0)
            temp_dict1[w1] += 1
        for w2 in s2:
            temp_dict2.setdefault(w2,0)
            temp_dict2[w2] += 1
        
        if num % 32 == 0:
            pb.display_progress_bar()
    sorted1 = sorted(temp_dict1.items(),key=lambda i:i[1],reverse=True)
    sorted2 = sorted(temp_dict2.items(),key=lambda i:i[1],reverse=True)
    #print(sorted1[:100])
    #print(sorted2[:100])
    return [i[0] for i in sorted1[:args.vac_dict_ch]],[i[0] for i in sorted2[:args.vac_dict_en]]
    
most_common_ch ,most_common_en = get_most_common(train_token_chinese,train_token_english)
print("\n ch words:{} en words:{}".format(len(most_common_ch),len(most_common_en)))
print(most_common_ch[:20])
print(most_common_en[:20])

print('zipping...')
ind2ch = dict(zip(range(1,len(most_common_ch) + 1),most_common_ch))
ch2ind = dict(zip(most_common_ch,range(1,len(most_common_ch) + 1)))
ind2en = dict(zip(range(1,len(most_common_en) + 1),most_common_en))
en2ind = dict(zip(most_common_en,range(1,len(most_common_en) + 1)))

print('toklizing...')
train_x = [[en2ind.get(j,0) for j in i] for i in train_token_english if  (args.max_words == -1 or len(i) < args.max_words)]
train_y = [[ch2ind.get(j,0) for j in i] for i in train_token_chinese if  (args.max_words == -1 or len(i) < args.max_words)]

print(len(train_x),len(train_y))
print(train_x[0])
print(' '.join([ind2en.get(i,'<unk>') for i in train_x[0]]))
print(train_y[0])
print(' '.join([ind2ch.get(i,'<unk>') for i in train_y[0]]))

with open('middleresult/tokenlizer_output_{}ch_{}en_{}words.pkl'.format(args.vac_dict_ch,args.vac_dict_en,args.max_words),'wb') as whdl:
    pickle.dump((
         ind2ch,
         ch2ind,
         ind2en,
         en2ind,
         train_x,
         train_y,
    ),whdl)