"""
Created: @August 1st, 2019

Description:
Get predicted labels from Pre-trained ERNIE classifier.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import paddle.fluid as fluid

from model.ernie import ErnieConfig
from finetune.classifier import create_model, predict
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser
from reader.task_reader import ClassifyReader

args = parser.parse_args()


# Todo
def ensemble_spo():
    '''
    receive: subj, obj, docid, relation_label
    :return: SPO dictionary
    '''

    return None



def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    # ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

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

    # Specify data reader
    reader = ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed
    )

    # Get data from reader
    examples = reader._read_tsv(args.test_set)

    subjects = [example.subject for example in examples]
    objects = [example.objects for example in examples]
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
    assert len(preds) == len(subjects) == len(objects)
    unique_docs = len(set(docids))

    for i in range(len(examples)):
        # Todo






if __name__ == '__main__':
    # print_arguments(args)
    main(args)
