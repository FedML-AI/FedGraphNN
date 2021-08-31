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

command = "kill $(ps aux | grep fed_subgraph_link_pred.py | grep -v grep | awk '{{print $2}}')"
print(command)
os.system(command)

# client_num_hpo = [8, 28] # 2
client_num_hpo = [8] # 2
dataset_hpo = ["epinions"] # 2
model_hpo = ["sage"] # 3
partition_alpha_hpo = [0.1]
round_num_hpo = [100] # 3
local_epoch_hpo = [1, 2, 5] # 3
batch_size_hpo = [1]
lr_hpo = [0.01, 0.001] # 3
# hidden_dim_hpo = [32, 64, 128, 256] # 4
hidden_dim_hpo = [64] # 4
n_layers_hpo = [3]
dropout_hpo = [0.5]

run_id = 0
for client_num in client_num_hpo:
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
                                            print(args.starting_run_id)
                                            print(run_id)
                                            if run_id < args.starting_run_id:
                                                run_id += 1
                                                continue

                                            args.client_num = client_num
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
                                            args.run_id = run_id
                                            if (
                                                args.dataset == "ciao"
                                                and args.client_num == 28
                                            ):
                                                args.client_num = 28
                                            elif (
                                                args.dataset == "epinions"
                                                and args.client_num == 28
                                            ):
                                                args.client_num = 27

                                            print(args)
                                            # sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 1 20 1 0.01 64 5 0.1 ciao
                                            os.system(
                                                "nohup sh run_fed_subgraph_link_pred.sh {args.client_num} {args.client_num} 1 8 {args.model} uniform {args.partition_alpha} "
                                                "{args.round_num} {args.epoch} {args.batch_size} {args.lr} {args.hidden_dim} {args.n_layers} {args.dr} {args.dataset} "
                                                "> ./fedgnn_rs_{args.run_id}.log 2>&1 &".format(
                                                    args=args
                                                )
                                            )
                                            wait_for_the_training_process()
                                            logging.info("cleaning the training...")
                                            command = "kill $(ps aux | grep fed_subgraph_link_pred.py | grep -v grep | awk '{{print $2}}')"
                                            print(command)
                                            os.system(command)
                                            sleep(5)
                                            run_id += 1
