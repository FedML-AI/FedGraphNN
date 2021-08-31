import argparse
import logging
import os
from time import sleep

"""
usage:
python3 sweep_rs.py --starting_run_id 0
"""


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # PipeTransformer related
    parser.add_argument("--starting_run_id", type=int, default=0)
    return parser.parse_args()


def wait_for_the_training_process():
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'" % message)
                print("Training is finished. Start the next training with...")
                os.remove(pipe_path)
                return
            sleep(3)
            print("Daemon is alive. Waiting for the training result.")


# customize the log format
logging.basicConfig(
    level=logging.INFO,
    format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)

parser = argparse.ArgumentParser()
args = add_args(parser)

command = "kill $(ps aux | grep fed_node_clf.py | grep -v grep | awk '{{print $2}}')"
print(command)
os.system(command)

# dataset_hpo = ["cora", "citeseer", "DBLP", "PubMed"]
dataset_hpo = ["cora"]
model_hpo = ["gcn"]
# model_hpo = ["gcn", "sgc", "sage"]
partition_alpha_hpo = [10.0]
round_num_hpo = [100]
local_epoch_hpo = [1]
batch_size_hpo = [1]
lr_hpo = [0.1, 0.01, 0.001]

# model
hidden_dim_hpo = [128]
n_layers_hpo = [3]
dropout_hpo = [0.5]
weight_decay_hpo = [1e-5]

run_id = 0

for dataset in dataset_hpo:
    for model in model_hpo:
        for partition_alpha in partition_alpha_hpo:
            for round_num in round_num_hpo:
                for epoch in local_epoch_hpo:
                    for batch_size in batch_size_hpo:
                        for lr in lr_hpo:
                            for hidden_dim in hidden_dim_hpo:
                                for n_layers in n_layers_hpo:
                                    for dr in dropout_hpo:
                                        for weight_decay in weight_decay_hpo:
                                            print(args.starting_run_id)
                                            print(run_id)
                                            if run_id < args.starting_run_id:
                                                run_id += 1
                                                continue

                                            args.dataset = dataset
                                            args.model = model
                                            args.partition_alpha = partition_alpha
                                            args.round_num = round_num
                                            args.epoch = epoch
                                            args.batch_size = batch_size
                                            args.lr = lr
                                            args.hidden_dim = hidden_dim
                                            args.n_layers = n_layers
                                            args.dr = dr
                                            args.weight_decay = weight_decay
                                            args.run_id = run_id

                                            print(args)
                                            # sh run_fed_node_clf.sh 10 10 1 1 gcn hetero 2.0 20 1 32 0.0015 32 3 0.3 cora
                                            os.system(
                                                "nohup sh run_fed_node_clf.sh 10 10 1 8 {args.model} hetero {args.partition_alpha} {args.round_num} "
                                                "{args.epoch} {args.batch_size} {args.lr} {args.hidden_dim} {args.n_layers} {args.dr} {args.weight_decay} {args.dataset} "
                                                "> ./fedgnn_ego_{args.run_id}.log 2>&1 &".format(
                                                    args=args
                                                )
                                            )
                                            wait_for_the_training_process()
                                            logging.info("cleaning the training...")
                                            command = "kill $(ps aux | grep fed_node_clf.py | grep -v grep | awk '{{print $2}}')"
                                            print(command)
                                            os.system(command)
                                            sleep(5)
                                            run_id += 1
