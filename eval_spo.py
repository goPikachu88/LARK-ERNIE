'''
Created: August 2nd, 2019

'''
import json
from pprint import pprint
from collections import Counter
from argparse import ArgumentParser, FileType


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--gold', type=FileType('r'), help='Original dataset')
    parser.add_argument('--system', type=FileType('r'), help='System predicted SPOs')
    parser.add_argument('--output', type=FileType('w'),
                        help='Save wrong predictions for error analysis.')

    return parser.parse_args()


def count_instances(actual, pred):
    TP = []
    FP = []

    '''
    Unpack the condition of SPO matching into 3 flags, in order to add/delete/modify rules conveniently
    '''
    def _match_predicate(gold_spo, sys_spo):
        return True if gold_spo['predicate'] == sys_spo['predicate'] else False

    # Todo: more inference rules
    def _match_subject(gold_spo, sys_spo):

        if gold_spo['subject'] == sys_spo['subject']:
            return True
        else:
            return False

    def _match_object(gold_spo, sys_spo):
        if gold_spo['object'] == sys_spo['object']:
            return True
        else:
            return False

    # Match & count
    for sys_spo in pred:
        flag_found = False

        for gold_spo in actual:
            if _match_predicate(gold_spo, sys_spo) \
                    and _match_subject(gold_spo, sys_spo) \
                    and _match_object(gold_spo, sys_spo):
                TP.append(sys_spo)
                flag_found = True
            else:
                continue

        if not flag_found:
            FP.append(sys_spo)
    # end for

    return TP, FP


def main():
    args = arg_parse()

    gold_docs = [json.loads(line) for line in args.gold]
    for i, d in enumerate(gold_docs):
        d.update({"docid": "doc-%d" % i})       # a temporary docid for alignining between gold & system
    system_docs = [json.loads(line) for line in args.system]
    # assert len(gold_docs) == len(system_docs)

    n_correct = 0
    n_actual = 0
    n_pred = 0

    # Eval doc by doc
    for i, g in enumerate(gold_docs):
        docid = "doc-%d" % i
        gold_spos = g.get('spo_list', [])
        n_actual += len(gold_spos)

        system_spos = []
        for s in system_docs:
            if s.get('docid', '') == docid:
                system_spos = s.get('spo_list', [])
        # end for
        n_pred += len(system_spos)

        tp, fp = count_instances(actual=gold_spos, pred=system_spos)
        n_correct += len(tp)
        if fp:
            args.output.write(json.dumps(dict(docid=docid, fp=fp),
                                         sort_keys=True,
                                         ensure_ascii=False)
                              )
            args.output.write("\n")
    # end for
    print("Wrong predictions saved to %s" % args.output.name)

    # Calculate P, R, F1
    precision = n_correct / n_pred
    recall = n_correct / n_actual
    f1 = 2 * precision * recall / (precision + recall) if n_correct else 0.0

    print("SPO Evaluation Results:")
    print("Precision: %.4f, Recall: %.4f, F1: %.4f" % (precision, recall, f1))


if __name__ == '__main__':
    main()
