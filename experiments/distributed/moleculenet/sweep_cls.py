import argparse
import logging
import os
from time import sleep

"""
put the following code before the finish() function at FedML/fedml_api/distributed/fedavg/FedAvgServerManager.py
                def post_complete_message_to_sweep_process(args):
                    logging.info("post_complete_message_to_sweep_process")
                    pipe_path = "./moleculenet_cls"
                    if not os.path.exists(pipe_path):
                        os.mkfifo(pipe_path)
                    pipe_fd = os.open(pipe_path, os.O_WRONLY)

                    with os.fdopen(pipe_fd, 'w') as pipe:
                        pipe.write("training is finished! \n%s" % (str(args)))

                post_complete_message_to_sweep_process(self.args)
                
"""


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # PipeTransformer related
    parser.add_argument("--starting_run_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="graphsage")
    parser.add_argument("--dataset", type=str, default="sider")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default="gpu")
    return parser.parse_args()


def wait_for_the_training_process():
    pipe_path = "./moleculenet_cls"
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

print(args)
command = "kill $(ps aux | grep main_fedavg.py | grep -v grep | awk '{{print $2}}')"
print(command)
os.system(command)
# os.system("kill $(ps aux | grep \"main_fedavg_graphsage_bace.py\" | grep -v grep | awk '{print $2}')")

# CLIENT_NUM=$1 -> Number of clients in dist/fed setting
# WORKER_NUM=$2 -> Number of workers
# SERVER_NUM=$3 -> Number of servers
# GPU_NUM_PER_SERVER=$4 -> GPU number per server
# MODEL=$5 -> Model name
# DISTRIBUTION=$6 -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
# ROUND=$7 -> Number of Distiributed/Federated Learning Rounds
# EPOCH=$8 -> Number of epochs to train clients' local models
# BATCH_SIZE=$9 -> Batch size
# LR=${10}  -> learning rate
# SAGE_DIM=${11} -> Dimenionality of GraphSAGE embedding
# NODE_DIM=${12} -> Dimensionality of node embeddings
# SAGE_DR=${13} -> Dropout rate applied between GraphSAGE Layers
# READ_DIM=${14} -> Dimensioanlity of readout embedding
# GRAPH_DIM=${15} -> Dimensionality of graph embedding
# DATASET=${16} -> Dataset name (Please check data folder to see all available datasets)
# DATA_DIR=${17} -> Dataset directory
# CI=${18}
# 3 * 3 * 4 * 3 * 3 * 3 * 3 * 3 = 8748
# 4 * 3 * 3 = 36
round_hop = [50]
epoch_hop = [1]
lr_hpo = [0.00015, 0.0015, 0.015, 0.15]
node_dim_hpo = [64]
sage_dim_hpo = [64]
read_dim_hpo = [64]
graph_dim_hpo = [64]
dropout_hpo = [0.3, 0.5, 0.6]

run_id = 0
args.run_id = run_id
for round in round_hop:
    for epoch in epoch_hop:
        for lr in lr_hpo:
            for node_dim in node_dim_hpo:
                for sage_dim in sage_dim_hpo:
                    for read_dim in read_dim_hpo:
                        for graph_dim in graph_dim_hpo:
                            for dr in dropout_hpo:
                                print(args.starting_run_id)
                                print(run_id)
                                if run_id < args.starting_run_id:
                                    run_id += 1
                                    continue
                                args.round = round
                                args.epoch = epoch
                                args.lr = lr
                                args.node_dim = node_dim
                                args.sage_dim = sage_dim
                                args.read_dim = read_dim
                                args.graph_dim = graph_dim
                                args.dr = dr
                                args.run_id = run_id
                                logging.info(args)

                                # sh run_fedavg_distributed_pytorch.sh 4 4 1 4 graphsage hetero 0.2 20 1 1 0.0015 64 32 0.3 64 64 sider "./../../../data/sider/" 0
                                os.system(
                                    "nohup sh run_fedavg_distributed_pytorch.sh 4 4 1 {args.gpu} "
                                    "{args.model_name} "
                                    "hetero "
                                    "{args.alpha} "
                                    "{args.round} "
                                    "{args.epoch} "
                                    "1 "
                                    "{args.lr} "
                                    "{args.sage_dim} "
                                    "{args.node_dim} "
                                    "{args.dr} "
                                    "{args.read_dim} "
                                    "{args.graph_dim} "
                                    "{args.dataset} "
                                    '"./../../../data/{args.dataset}/" 0 '
                                    "> ./fedgnn_cls_{args.model_name}_{args.dataset}_{args.run_id}.log 2>&1 &".format(
                                        args=args
                                    )
                                )
                                wait_for_the_training_process()
                                # sleep(3600*10)
                                logging.info("cleaning the training...")
                                command = "kill $(ps aux | grep main_fedavg.py | grep -v grep | awk '{{print $2}}')"
                                print(command)
                                os.system(command)
                                # os.system("kill $(ps aux | grep main_fedavg.py | grep -v grep | awk {print $2})".format(args=args))
                                sleep(5)
                                run_id += 1
