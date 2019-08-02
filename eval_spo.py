'''

Created: August 2nd, 2019




'''
import json
from pprint import pprint
from argparse import ArgumentParser, FileType



def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--gold', type=FileType('r'), help='Original dataset')
    parser.add_argument('--system', type=FileType('r'), help='System predicted SPOs')
    parser.add_argument('--output', type=FileType('w'),
                        help='Evaluate report.')

    return parser.parse_args()


def main():
    args = arg_parse()

    gold_docs = [json.loads(line) for line in args.gold]
    for i, d in enumerate(gold_docs):
        d.update({"docid": "dev-%d" % i})
    system_docs = [json.loads(line) for line in args.system]
    assert len(gold_docs) == len(system_docs)

    n_correct = 0
    n_actual = 0
    n_pred = 0

    # Eval doc by doc
    for i, d in gold_docs:
        docid = "dev-%d" % i

        gold_spos = d.get('spo_list', [])
        n_actual += len(gold_spos)

        sys_doc =




if __name__ == '__main__':
    main()
