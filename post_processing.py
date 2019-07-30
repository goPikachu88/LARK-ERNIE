'''
Created: July 30, 2019

Description:
Post-process the NER results of ERNIE. Produce datasets for ERNIE RE.

Steps:
1. Extract entities from the row-wise NER result file
2. Add ['entity_list'] filed for each document in json file
    entity_dict(text= ,
                type= ,
                # role= )
3. Apply RelationTransformer to generate data for RE prediction


Format:
    Char	Actual	Pred
    #doc-0
    查	B-人物	B-人物
    尔	I-人物	I-人物
    斯	I-人物	I-人物
    [UNK]	I-人物	I-人物
    阿	I-人物	I-人物
    兰	I-人物	I-人物
    基	I-人物	I-人物
    斯	I-人物	I-人物
    （	O	O
    charles	O	O
    ar	O	O
    ##ang	O	O
    ##ui	O	O
    ##z	O	O
    ）	O	O
    ，	O	O

Desired Output:


'''

import os
import json
from collections import namedtuple
from preprocess import RelationTransformer





def extract_entities(doc):
    '''
    Input format:
        {'docid': 'dev-0',
        'lines': ['查\tB-人物\tB-人物', '尔\tI-人物\tI-人物', '斯\tI-人物\tI-人物', '[UNK]\tI-人物\tI-人物', '阿\tI-人物\tI-人物', '兰\tI-人物\tI-人物', '基\tI-人物\tI-人物', '斯\tI-人物\tI-人物', '（\tO\tO', 'charles\tO\tO', 'ar\tO\tO', '##ang\tO\tO', '##ui\tO\tO', '##z\tO\tO', '）\tO\tO', '，\tO\tO', '1989\tB-Date\tB-Date', '年\tI-Date\tI-Date', '4\tI-Date\tI-Date', '月\tI-Date\tI-Date', '17\tI-Date\tI-Date', '日\tI-Date\tI-Date', '出\tO\tO', '生\tO\tO', '于\tO\tO', '智\tO\tB-国家', '利\tO\tI-国家', '圣\tB-地点\tB-地点', '地\tI-地点\tI-地点', '亚\tI-地点\tI-地点', '哥\tI-地点\tI-地点']}

    Output format:
        {'docid': 'dev-0',
        'text': str,
        'gold_entity_list': [],
        'system_entity_list': []
        }
    '''
    lines = doc.get('lines', [])

    Example = namedtuple('Example', ['text', 'actual', 'pred'])
    lines = [line.split('\t') for line in lines]
    print(lines[0])
    converted = [Example(*line) for line in lines]
    print(converted[:3])

#     entity_list =

#     return entity_list


def _read_file(fpath):
    f = open(fpath, 'r').readlines()
    headers = f[0].rstrip().split('\t')
    lines = [line.rstrip() for line in f[1:]]

    doc_starts = [i for i, line in enumerate(lines) if line.startswith("#dev-")]
    print(len(doc_starts))
    doc_starts.append(len(lines))       # add end

    docs = []
    for i in range(len(doc_starts)-1):
        start= doc_starts[i]
        end = doc_starts[i+1]
        docid = lines[start].strip("#")
        doc_lines = [line for line in lines[start+1:end] if line not in ['\n', '']]     # remove blank rows
        docs.append(dict(docid=docid,
                         lines = doc_lines))

    return docs



def main():
    docs = _read_file("/home/yue/Desktop/dev_ner_all.txt")
    extract_entities(docs[0])



if __name__ == '__main__':
    main()
