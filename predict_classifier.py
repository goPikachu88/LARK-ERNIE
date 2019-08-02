"""
Created: @August 1st, 2019

Description:
Get predicted labels from Pre-trained ERNIE classifier.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import paddle.fluid as fluid

from model.ernie import ErnieConfig
from finetune.classifier import create_model, predict
from utils.args import print_arguments
from utils.init import init_checkpoint
from finetune_args import parser
from reader.task_reader import ClassifyReader

args = parser.parse_args()


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Specify data reader
    reader = ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed
    )

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    main_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            test_pyreader, graph_vars = create_model(args,
                                                     pyreader_name='predict_reader',
                                                     ernie_config=ernie_config
                                                     )
    main_prog = main_prog.clone(for_test=True)

    exe.run(startup_prog)

    # Load model
    if not args.init_checkpoint:
        raise ValueError("args 'init_checkpoint' should be set if only doing prediction!")
    init_checkpoint(
        exe,
        args.init_checkpoint,
        main_program=startup_prog,
        use_fp16=args.use_fp16
    )


    # Get data from reader
    examples = reader._read_tsv(args.test_set)

    subjects = [example.subject for example in examples]
    objects = [example.object for example in examples]
    docids = [example.docid for example in examples]

    # Predict
    test_pyreader.decorate_tensor_provider(
        reader.data_generator(
            args.test_set,
            batch_size=args.batch_size,
            epoch=1,
            shuffle=False))     # shuffle must set to false to keep the order
    preds = predict(exe, main_prog, test_pyreader, graph_vars)


    # Write SPO results
    assert len(preds) == len(examples)
    unique_docids = sorted(list(set(docids)))

    LABEL2ID_MAP = json.load(open(args.label_map_config, "r"))
    ID2LABEL_MAP = {v: k for k, v in LABEL2ID_MAP.items()}


    # Assemble SPO by document
    results = []
    for _id in unique_docids:
        indices = [i for i,e in enumerate(examples) if e.docid == _id]

        spo_list = []
        for i in indices:
            relation = ID2LABEL_MAP[preds[i]]
            if relation != 'no_relation':
                spo = dict(predicate = relation,
                           subject = subjects[i],
                           object = objects[i])
                spo_list.append(spo)
        # end for

        results.append(dict(docid=_id, spo_list=spo_list))
    # end for


    # Dump SPO results json
    f_output = '/home/yue/Desktop/dev_predicted.json'
    with open(f_output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False))
            f.write('\n')
    print('File written to %s' % f_output)



if __name__ == '__main__':
    print_arguments(args)
    main(args)
