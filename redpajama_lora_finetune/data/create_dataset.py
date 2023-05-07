import csv 
import json

funny_data = []
funny_data_eval = []
csv_file = open('reddit_dadjokes.csv', mode='r')
csv_reader = csv.reader(csv_file)
i = 0
for row in csv_reader:
    if 'http' in row[2] or 'imgur' in row[2]:
        continue
    i += 1
    #print(row[2])
    print(i)
    line = {"input": "write me a good dad joke", "output":row[2]}

    if i%500 == 0:
        funny_data_eval.append(line)
    else:
        funny_data.append(line)
    # print('---------------------------------------------------------')
    # if i > 100:
    #     break

with open('train.jsonl', mode='w') as jsonl_file:
    for item in funny_data:
        json_line = json.dumps(item) 
        jsonl_file.write(json_line + '\n')

with open('eval.jsonl', mode='w') as jsonl_file:
    for item in funny_data_eval:
        json_line = json.dumps(item) 
        jsonl_file.write(json_line + '\n')