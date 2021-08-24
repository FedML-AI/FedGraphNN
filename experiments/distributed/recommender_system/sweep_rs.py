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
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

print(args)
command = "kill $(ps aux | grep main_fedavg.py | grep -v grep | awk '{{print $2}}')"
print(command)
os.system(command)
# os.system("kill $(ps aux | grep \"main_fedavg_graphsage_bace.py\" | grep -v grep | awk '{print $2}')")

partition_alpha_hpo = [0.1, 0.5, 1.0, 10.0]
rounds_hpo = [10, 20, 50, 100, 200]
local_epoch_hpo = [1, 2, 5]
batch_size_hpo = [1]
lr_hpo = [0.1, 0.01, 0.001, 0.0001]
hidden_dim_hpo = [32, 64, 128]
n_layers_hpo = [3, 5, 8, 10]
dropout_hpo = [0.1, 0.3, 0.5, 0.6]

run_id = 0
args.run_id = run_id
for partition_alpha in partition_alpha_hpo:
    for round in rounds_hpo:
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

                                # sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 1 20 1 0.01 64 5 0.1 ciao
                                os.system('nohup sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 1 20 1 0.01 64 5 0.1 ciao '
                                          '> ./fedgnn_cls_{args.model_name}_{args.dataset}_{args.run_id}.log 2>&1 &'.format(args=args))
                                wait_for_the_training_process()
                                # sleep(3600*10)
                                logging.info("cleaning the training...")
                                command = "kill $(ps aux | grep main_fedavg.py | grep -v grep | awk '{{print $2}}')"
                                print(command)
                                os.system(command)
                                # os.system("kill $(ps aux | grep main_fedavg.py | grep -v grep | awk {print $2})".format(args=args))
                                sleep(5)
                                run_id += 1
