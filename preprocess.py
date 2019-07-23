'''
Created: July 19, 2019

Description:
Pre-process datasets to satisfy the format of ERNIE.

Todo:
Label_map is defined but actually not used for pre-processing step. Remove later.

'''

import csv
import json
import random
from pprint import pprint
from argparse import ArgumentParser, FileType
from itertools import combinations, permutations

from tokenization import FullTokenizer


class BaseTransformer(object):
    # def __init__(self, label_map):
    #     self.label_map = label_map

    def transform(self, data):
        '''
        Unpack json data and transform to target instances.
        '''
        # print("Json instances: %d" % len(data))
        #
        # # unpack spo_list
        # unpacked = []
        # for d in data:
        #     for spo in d['spo_list']:
        #         unpacked.append(dict(text=d['text'],
        #                              relation=spo['predicate'],
        #                              subj=spo['subject'],
        #                              obj=spo['object'],
        #                              # subj_type=self.label_map[spo['subject_type']],
        #                              # obj_type=self.label_map[spo['object_type']],
        #                              subj_type=spo['subject_type'],
        #                              obj_type=spo['object_type'],
        #                              )
        #                         )
        # print('Unpacked SPO: %d' % len(unpacked))
        # random.Random(4).shuffle(unpacked)
        #
        # # return [self._transform_one(d) for d in unpacked]
        # transformed = []
        # error_idx = []
        # for i, d in enumerate(unpacked):
        #     if i % 10000 == 0:
        #         print('Processed %d unpacked examples.' % i)
        #     try:
        #         o = self._transform_one(d)
        #     except:
        #         o = {}          # maybe more robust to skip the instance than to return {}?
        #         error_idx.append(i)
        #     transformed.append(o)
        # print('%d examples cannot be transformed.' % len(error_idx))
        # print(error_idx)
        # return transformed
        raise NotImplementedError('Method not implemented.')


    def _transform_one(self, instance):
        '''transform an unpacked spo instance from
            ['text'], ['subj'], ['obj'], ['relation'], ['subj_type'], ['obj_type']
        to
            ['text_a'], ['label']
        '''
        raise NotImplementedError('Method not implemenetd.')


class RelationTransformer(object):
    def __init__(self, relation_mask='[MASK]', ordered=False, none_label='no_relation', **kwargs):
        super().__init__(**kwargs)
        self.relation_mask = relation_mask
        self.ordered = ordered
        self.none_label = none_label


    def _transform_one(self, instance):
        '''
        Transform one document into multiple relation instances:
            1. Generate entity pairs for each document
            2. For each pair of entities:
                Add relation label;
                Replace the entity texts with '[MASK]'. Keep the contexts.
        '''
        if self.ordered:
            def combination_func(L): return permutations(L, 2)
        else:
            def combination_func(L): return combinations(L, 2)

        entity_list = instance.get('entity_list', [])
        relations = []
        for e1, e2 in combination_func(entity_list):
            print(e1['text'], e2['text'])
            text = instance.get('text', None)
            text = text.replace(e1['text'], self.relation_mask)
            text = text.replace(e2['text'], self.relation_mask)

            for spo in instance.get('spo_list', []):
                match = e1['text'] == spo['subject'] and e2['text'] == spo['object']
                reverse_match = e1['text'] == spo['object'] and e2['text'] == spo['subject']

                if match or reverse_match:
                    r = dict(text_a = text,
                             label = spo['predicate'])
                else:
                    r = dict(text_a=text,
                             label=self.none_label)
                relations.append(r)
            # end for
        # end for

        return relations


    def transform(self, instances):
        '''
        Transform a list of json instances (documents) into a list of relation instances.
        '''
        print("Json instances: %d" % len(instances))

        # return [self._transform_one(d) for d in documents]
        relations = []
        for i, d in enumerate(instances):
            if i % 10000 == 0:
                print('Processed %d documents.' % i)
                relations += self._transform_one(d)
        print('Relation instances (all): %d' % len(relations))

        # count positive percent
        n_pos = len([r for r in relations if r['label'] != self.none_label])
        print('Relation instances (positive): %d (%.2f%%)' % (n_pos, n_pos/len(relations)*100))

        random.Random(4).shuffle(relations)

        return relations


class NERTransformer(BaseTransformer):
    def __init__(self, sep=u"", **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = FullTokenizer(vocab_file='config/vocab.txt',
                                       do_lower_case=True)
        self.sep = sep

    def _transform_one(self, instance):
        '''
        Tokenize a piece of text and generate a sequence of NER labels for tokens.
        '''
        text_tokens = self.tokenizer.tokenize(instance['text'])
        subj_tokens = self.tokenizer.tokenize(instance['subj'])
        obj_tokens = self.tokenizer.tokenize(instance['obj'])
        labels = ["O"] * len(text_tokens)

        # Find entity boundaries
        subj_start, subj_end = self._find_sublist_boundary(subj_tokens, text_tokens)
        obj_start, obj_end = self._find_sublist_boundary(obj_tokens, text_tokens)

        # Add entity BIO labels
        # 65 labels
        # labels[subj_start] = 'B-%s-subj' % instance['subj_type']
        # labels[subj_start+1 : subj_end+1] = ['I-%s-subj' % instance['subj_type']] * (len(subj_tokens)-1)
        # labels[obj_start] = 'B-%s-obj' % instance['obj_type']
        # labels[obj_start+1 : obj_end+1] = ['I-%s-obj' % instance['obj_type']] * (len(obj_tokens)-1)

        # 57 labels
        labels[subj_start] = 'B-%s' % instance['subj_type']
        labels[subj_start+1 : subj_end+1] = ['I-%s' % instance['subj_type']] * (len(subj_tokens)-1)
        labels[obj_start] = 'B-%s' % instance['obj_type']
        labels[obj_start+1 : obj_end+1] = ['I-%s' % instance['obj_type']] * (len(obj_tokens)-1)

        # print('subj:', text_tokens[subj_start:subj_end+1], labels[subj_start:subj_end+1])
        # print('obj:', text_tokens[obj_start:obj_end+1], labels[obj_start:obj_end+1])

        return dict(text_a=self.sep.join(text_tokens),
                    label=self.sep.join(labels))

    def _find_sublist_boundary(self, sublist, full_list):
        '''
        Todo: A few instances cannot find sublist boundary.
        '''
        for start in (i for i, v in enumerate(full_list) if v==sublist[0]):
            if full_list[start : start+len(sublist)] == sublist:
                return (start, start+len(sublist)-1)
    # end def



def transform_json(instances):
    '''transform an unpacked spo instance from
            ['text'], ['spo_list'], ['postag']
        to
            ['text'], ['spo_list'], ['docid'], ['entity_list']
    '''
    for i, ins in enumerate(instances):
        ins.pop('postag', None)     # remove ['postag']
        ins['docid'] = 'doc_%s' % i

        # unpack entity
        entities = []
        spo_list = ins.get('spo_list', [])
        for spo in spo_list:
            entities.append(dict(text=spo['subject'], type=spo['subject_type'], role='subject'))
            entities.append(dict(text=spo['object'], type=spo['object_type'], role='object'))

        entities = [dict(e) for e in set([tuple(e.items()) for e in entities])]     # remove duplicates
        ins['entity_list'] = entities
    # end for

    return instances


def write2tsv(f, unpacked):
    writer = csv.DictWriter(f, fieldnames=['label', 'text_a'], delimiter='\t')
    writer.writeheader()
    for d in unpacked:
        writer.writerow(d)
    # end for
    print('File written to %s' % f.name)


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data', type=FileType('r'), help='Input json file')
    parser.add_argument('--output', type=FileType('w'), help='Output tsv file')
    return parser.parse_args()


def main():
    args = arg_parse()

    data_original = [json.loads(line) for line in args.data][:10]
    instances = transform_json(data_original)

    transformer = RelationTransformer()
    # transformer = NERTransformer(label_map = None)

    transformed = transformer.transform(instances)
    pprint(transformed)
    # write2tsv(f = args.output, unpacked = transformed)



if __name__ == '__main__':
    main()
