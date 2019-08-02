#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel


def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1],
                [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def calculate_metrics(y_true, y_pred, labels):
    micro_r = recall_score(y_true, y_pred, average='micro', labels=labels)
    micro_p = precision_score(y_true, y_pred, average='micro', labels=labels)

    macro_r = recall_score(y_true, y_pred, average='macro', labels=labels)
    macro_p = precision_score(y_true, y_pred, average='macro', labels=labels)

    if not micro_p or not macro_p:
        micro_f, macro_f = 0.0, 0.0
    else:
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
        macro_f = 2 * macro_p * macro_r / (macro_p + macro_r)

    metrics = dict(micro_r=micro_r, micro_p=micro_p, micro_f=micro_f,
                   macro_r=macro_r, macro_p=macro_p, macro_f=macro_f)
    return metrics


def evaluate(exe, test_program, test_pyreader, graph_vars, eval_phase):

    target_labels = list(range(49))     # excluding 'no_relation'

    # ******************     Training Phase    ************************
    if eval_phase == "train":
        train_fetch_list = [
            graph_vars["loss"].name,
            graph_vars["accuracy"].name,
            graph_vars["probs"].name,
            graph_vars["labels"].name
        ]
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)

        outputs = exe.run(fetch_list=train_fetch_list)

        preds = np.argmax(outputs[2], axis=1).astype(np.int8).tolist()
        labels = outputs[3].reshape((-1)).tolist()
        metrics = calculate_metrics(y_true=labels, y_pred=preds, labels = target_labels)

        ret = {"loss": np.mean(outputs[0]),
               "accuracy": np.mean(outputs[1]),
               "micro_p": metrics["micro_p"],
               "micro_r": metrics["micro_r"],
               "micro_f": metrics["micro_f"],
               "macro_p": metrics["macro_p"],
               "macro_r": metrics["macro_r"],
               "macro_f": metrics["macro_f"],
               }
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[-1][0])

        return ret


    # **************     Evaluation Phase    **************************
    test_pyreader.start()
    time_begin = time.time()

    total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    preds, labels, scores = [], [], []
    current_batch = 0

    fetch_list = [
        graph_vars["loss"].name,
        graph_vars["accuracy"].name,
        graph_vars["probs"].name,
        graph_vars["labels"].name,
        graph_vars["num_seqs"].name
    ]
    while True:     # evaluate by batch
        try:
            batch_time_begin = time.time()

            np_loss, np_acc, np_probs, np_labels, np_num_seqs = exe.run(
                program=test_program,
                fetch_list=fetch_list)
            np_preds = np.argmax(np_probs, axis=1).astype(np.int8)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)

            labels.extend(np_labels.reshape((-1)).tolist())
            preds.extend(np_preds.tolist())

            if current_batch % 5000 == 0:
                print('batch %d, elapsed time: %f' % (current_batch, time.time()-batch_time_begin))
            current_batch += 1

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    # calculate p, r f1
    metrics = calculate_metrics(y_true=labels, y_pred=preds, labels=target_labels)

    print(
        "[%s evaluation] ave loss: %f, ave_acc: %f, macro_p: %f, macro_r: %f, macro_f: %f, micro_p: %f, micro_r: %f, micro_f: %f, data_num: %d, elapsed time: %f s"
        % (eval_phase,
           total_cost / total_num_seqs,
           total_acc / total_num_seqs,
           metrics['macro_p'], metrics['macro_r'], metrics['macro_f'],
           metrics['micro_p'], metrics['micro_r'], metrics['micro_f'],
           total_num_seqs,
           time_end - time_begin))


def predict(exe, test_program, test_pyreader, graph_vars):

    fetch_list = [
        graph_vars["probs"].name,
        graph_vars["labels"].name,
        graph_vars["num_seqs"].name
    ]

    total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0, 0.0
    preds, labels = [], []
    current_batch = 0

    test_pyreader.start()
    time_begin = time.time()
    while True:         # evaluate by batch
        try:
            batch_time_begin = time.time()

            np_probs, np_labels, np_num_seqs = exe.run(
                program=test_program,
                fetch_list=fetch_list)
            np_preds = np.argmax(np_probs, axis=1).astype(np.int8)
            total_num_seqs += np.sum(np_num_seqs)

            labels.extend(np_labels.reshape((-1)).tolist())
            preds.extend(np_preds.tolist())

            if current_batch % 1000 == 0:
                print('batch %d, elapsed time: %f' % (current_batch, time.time() - batch_time_begin))
            current_batch += 1

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    print("Prediction completed, elapsed time: %f s" % (time_end - time_begin))

    # Todo: return preds with texts ?
    return preds

