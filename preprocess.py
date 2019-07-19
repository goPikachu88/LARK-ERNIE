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
from tokenization import FullTokenizer


class BaseTransformer(object):
    def __init__(self, label_map):
        # self.label_map = json.load(open(label_map, 'r'))
        self.label_map = label_map

    def transform(self, data):
        '''
        Unpack json data and transform to target instances.
        '''
        print("Json instances: %d" % len(data))

        # unpack spo_list
        unpacked = []
        for d in data:
            for spo in d['spo_list']:
                unpacked.append(dict(text=d['text'],
                                     relation=spo['predicate'],
                                     subj=spo['subject'],
                                     obj=spo['object'],
                                     subj_type=spo['subject_type'],
                                     obj_type=spo['object_type']))
        print('Unpacked SPO: %d' % len(unpacked))
        random.Random(4).shuffle(unpacked)

        return [self._transform_one(d) for d in unpacked]


    def _transform_one(self, instance):
        '''transform an unpacked spo instance from
            ['text'], ['subj'], ['obj'], ['relation'], ['subj_type'], ['obj_type']
        to
            ['text_a'], ['label']
        '''
        raise NotImplementedError('Method not implemenetd.')


class RelationTransformer(BaseTransformer):
    def __init__(self, relation_mask='[MASK]', **kwargs):
        super().__init__(**kwargs)
        self.relation_mask = relation_mask


    def _transform_one(self, instance):
        '''
        Replace the entity texts with '[MASK]'. Keep the contexts.
        '''
        text = instance['text']
        text = text.replace(instance['subj'], self.relation_mask)
        text = text.replace(instance['obj'], self.relation_mask)

        return dict(text_a=text,
                    # label=self.label_map.get(instance['relation'], 0),
                    label=instance['relation'],
                    )

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
        labels[subj_start] = 'B-%s-subj' % instance['subj_type']
        labels[subj_start+1 : subj_end+1] = ['I-%s-subj' % instance['subj_type']] * (len(subj_tokens)-1)
        labels[obj_start] = 'B-%s-obj' % instance['obj_type']
        labels[obj_start+1 : obj_end+1] = ['I-%s-obj' % instance['obj_type']] * (len(obj_tokens)-1)

        # print('subj:', text_tokens[subj_start:subj_end+1], labels[subj_start:subj_end+1])
        # print('obj:', text_tokens[obj_start:obj_end+1], labels[obj_start:obj_end+1])

        return dict(text_a=self.sep.join(text_tokens),
                    label=self.sep.join(labels))

    def _find_sublist_boundary(self, sublist, full_list):
        for start in (i for i, v in enumerate(full_list) if v==sublist[0]):
            if full_list[start : start+len(sublist)] == sublist:
                return (start, start+len(sublist)-1)
    # end def


def write2tsv(f, unpacked):
    writer = csv.DictWriter(f, fieldnames=['label', 'text_a'], delimiter='\t')
    writer.writeheader()
    for d in unpacked:
        writer.writerow(d)
    # end for
    print('File written to %s' % f.name)


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data', type=FileType('r'),
                        default='/home/yue/Data/ernie_processed/dev_data_postag.json',
                        help='Input json file')
    parser.add_argument('--label_map', type=FileType('r'), default=None,
                        help='Label map json file.')
    parser.add_argument('--output', type=FileType('w'), default='/home/yue/Desktop/test.tsv',
                        help='Output tsv file')
    return parser.parse_args()


def main():
    args = arg_parse()

    data_original = [json.loads(line) for line in args.data][:5]
    # transformer = RelationTransformer(label_map = json.load(args.label_map))
    transformer = NERTransformer(label_map = json.load(args.label_map))

    transformed = transformer.transform(data_original)

    pprint(transformed)
    write2tsv(f = args.output, unpacked = transformed)



if __name__ == '__main__':
    main()
