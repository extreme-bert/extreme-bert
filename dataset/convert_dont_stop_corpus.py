import jsonlines
import os
import csv

tasks = ['amazon', 'chemprot', 'hyperpartisan_news',  'imdb', 'rct-20k', 'sciie', 'ag', 'citation_intent']
PATH_TO_DONT_STOP_DATA = ""
PATH_TO_SAVE_DATA = ""
for task in tasks:
    target_path = f"{PATH_TO_SAVE_DATA}/{task}/"

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    f_write = open(f"{target_path}/train.tsv", "w")
    tsv_w = csv.writer(f_write, delimiter='\t')

    labels = set()
    with open(f'{PATH_TO_DONT_STOP_DATA}/{task}/train.jsonl', 'rb') as f:
        for item in jsonlines.Reader(f):
            item['text'] = item['text'].replace('\n', ' ')  #only for amazon and imdb because there are \n in amazon and imdb
            tsv_w.writerow([item['text'], item['label']])
            labels.add(item['label'])
    print('all labels: ', labels)
    print('number of labels: ', len(labels))
    f_write.close()