'''
Created: July 19, 2019

Description:
Pre-process datasets to satisfy the format of ERNIE.


'''

import csv
import json
import random
from pprint import pprint
from argparse import ArgumentParser, FileType
from itertools import combinations, permutations

from tokenization import FullTokenizer


class RelationTransformer(object):
    def __init__(self, relation_mask='[MASK]', ordered=True, none_label='no_relation', downsample=False,  **kwargs):
        super().__init__(**kwargs)
        self.relation_mask = relation_mask
        self.ordered = ordered
        self.none_label = none_label
        self.downsample = downsample


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
        for subj, obj in combination_func(entity_list):
            text = instance.get('text', None)
            text = text.replace(subj['text'], "%s%s" % (self.relation_mask, self.relation_mask))
            text = text.replace(obj['text'], self.relation_mask)

            r = None
            for spo in instance.get('spo_list', []):
                match = subj['text'] == spo['subject'] and obj['text'] == spo['object']

                if match:
                    r = dict(text_a = text,
                             label = spo['predicate'])
                    break
            # end for
            if not r:
                r = dict(text_a=text,
                         label=self.none_label)
            relations.append(r)
            # print(subj['text'], obj['text'], r['label'])
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

        # count positive percent
        print('Relation instances (all): %d' % len(relations))
        n_pos = len([r for r in relations if r['label'] != self.none_label])
        print('Relation instances (positive): %d (%.2f%%)' % (n_pos, n_pos/len(relations)*100))

        if self.downsample:
            relations = self._downsample_negative(relations)
        random.Random(4).shuffle(relations)

        return relations


    def _downsample_negative(self, instances):
        positive = [ins for ins in instances if ins['label'] != self.none_label]
        negative = [ins for ins in instances if ins['label'] == self.none_label]

        selected = random.Random(4).sample(negative, len(positive))        # pos:neg = 1:1
        selected += positive
        assert len(selected) == len(positive)*2

        # random.Random(4).shuffle(selected)
        print('Downsampled to POS:NEG=1:1')

        return selected


class NERTransformer(object):
    '''
    Input: a dict with ['text'], ['spo_list'], ['docid'] and ['entity_list']
    Output: tsv file with ['text_a'], ['labels'], tokens separated by u"". Keep the doc order as in json.

    Todo: add docid?
    '''
    def __init__(self, sep=u"", do_lower_case=True):
        self.tokenizer = FullTokenizer(vocab_file='config/vocab.txt',
                                       do_lower_case=do_lower_case)
        self.sep = sep

    def _transform_one(self, instance):
        '''
        Tokenize a piece of text and generate a sequence of NER labels for tokens.
        '''
        text_tokens = self.tokenizer.tokenize(instance['text'])
        labels = ["O"] * len(text_tokens)

        entities = instance.get('entity_list', [])
        n_overlap = 0
        for e in entities:
            e_tokens = self.tokenizer.tokenize(e['text'])
            try:
                e_start, e_end = self._find_sublist_boundary(e_tokens, text_tokens)
            except:
                continue

            # if the span already labelled, skip
            if len(set(labels[e_start : e_end+1])) > 1:
                # print('Overlap.')
                n_overlap += 1
                continue

            # Add entity BIO labels (57 labels)
            labels[e_start] = 'B-%s' % e['type']
            labels[e_start+1 : e_end+1] = ['I-%s' % e['type']] * (len(e_tokens)-1)
            # print('entity:', text_tokens[e_start:e_end+1], labels[e_start:e_end+1])

        return n_overlap, dict(text_a=self.sep.join(text_tokens), label=self.sep.join(labels))

    def _find_sublist_boundary(self, sublist, full_list):
        '''
        Todo: A few instances cannot find sublist boundary.
        '''
        for start in (i for i, v in enumerate(full_list) if v==sublist[0]):
            if full_list[start : start+len(sublist)] == sublist:
                return (start, start+len(sublist)-1)
    # end def

    def transform(self, instances):
        '''
        Transform a list of json instances (documents) into a list of NER instances.
        '''
        print("Json instances: %d" % len(instances))

        transformed = []
        n_total, n_overlap = 0, 0
        for i, d in enumerate(instances):
            if i % 10000 == 0:
                print('Processed %d documents.' % i)
            n_total += len(d.get('entity_list', []))
            n_overlap += self._transform_one(d)[0]
            transformed.append(self._transform_one(d)[1])

        print('(Entities) Total: %d, Overlapped: %d, Labelled: %d' % (n_total, n_overlap, n_total-n_overlap))

        return transformed


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

    data_original = [json.loads(line) for line in args.data]
    instances = transform_json(data_original)

    # transformer = RelationTransformer(downsample=True)
    transformer = NERTransformer()

    transformed = transformer.transform(instances)
    # pprint(transformed)
    # write2tsv(f = args.output, unpacked = transformed)


if __name__ == '__main__':
    main()
