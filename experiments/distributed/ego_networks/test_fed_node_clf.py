import argparse
import os
import socket
import sys

import psutil
import setproctitle
import torch.nn
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
#from data_preprocessing.ego_networks.data_loader import *
from test_data_loader import *
from model.ego_networks.gcn import GCNNodeCLF
from model.ego_networks.sgc import SGCNodeCLF
from model.ego_networks.sage import SAGENodeCLF

from training.ego_networks.federated_nc_trainer import FedNodeClfTrainer

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init

from experiments.distributed.initializer import (
    add_federated_args,
    get_fl_algorithm_initializer,
    set_seed,
)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CS",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/ego-networks",
        help="data directory",
    )

    parser.add_argument(
        "--part_file",
        type=str,
        default="./../../../data/ego-networks/partition",
        help="File name for predefined partitions in pickle format",
    )

    parser.add_argument(
        "--ego_number", type=int, default=1000, help="Sampled ego nodes"
    )

    parser.add_argument("--hop_number", type=int, default=5, help="Number of hops")

    parser.add_argument(
        "--normalize_features",
        type=bool,
        default=False,
        help="Whether or not to symmetrically normalize feat matrices",
    )

    parser.add_argument(
        "--normalize_adjacency",
        type=bool,
        default=False,
        help="Whether or not to symmetrically normalize adj matrices",
    )

    parser.add_argument(
        "--sparse_adjacency",
        type=bool,
        default=False,
        help="Whether or not the adj matrix is to be processed as a sparse matrix",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    # model related
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Size of GraphSAGE hidden layer"
    )

    parser.add_argument(
        "--n_layers", type=int, default=5, help="Number of GraphSAGE hidden layers"
    )

    parser.add_argument(
        "--node_embedding_dim",
        type=int,
        default=32,
        help="Dimensionality of the vector space the atoms will be embedded in",
    )

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha value for LeakyRelu used in GAT"
    )

    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of attention heads used in GAT"
    )

    parser.add_argument(
        "--K", type=int, default=32, help="Size of GraphSAGE hidden layer"
    )

    parser.add_argument(
        "--eps", type=int, default=0, help="Epsilon parameter used in GIN"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used between GraphSAGE layers",
    )

    parser.add_argument(
        "--graph_embedding_dim",
        type=int,
        default=64,
        help="Dimensionality of the vector space the molecule will be embedded in",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument(
        "--gpu_num_per_server", type=int, default=8, help="gpu_num_per_server"
    )

    parser = add_federated_args(parser)
    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    num_cats, feat_dim = 0, 0
    if args.dataset not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        raise Exception("no such dataset!")
    elif args.dataset in ["CS", "Physics"]:
        args.type_network = "coauthor"
    else:
        args.type_network = "citation"

    compact = args.model == "graphsage"

    unif = True if args.partition_method == "homo" else False

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    _, _, feat_dim, num_cats = get_data(args.data_dir, args.dataset)

    (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ) = load_partition_data(
        args,
        args.data_dir,
        args.client_num_in_total,
        uniform=unif,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ]

    return dataset, num_cats, feat_dim


def create_model(args, model_name, feat_dim, num_cats, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, num_cats)
    )
    if model_name == "gcn":
        model = GCNNodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
    elif model_name == "sgc":
        model = SGCNodeCLF(in_dim=feat_dim, num_classes=num_cats, K=args.n_layers)
    elif model_name == "sage":
        model = SAGENodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
    else:
        # MORE MODELS
        raise Exception("such model does not exist !")
    trainer = FedNodeClfTrainer(model)
    logging.info("Model and Trainer  - done")
    return model, trainer


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # device = torch.device("cuda:" + str(gpu_num_per_machine))
    # # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(process_gpu_dict[process_ID - 1])
    device = torch.device(
        "cuda:" + str(process_gpu_dict[process_ID - 1])
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(device)
    return device


if __name__ == "__main__":
    #     # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    #     # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    #     # customize the process name
    str_process_name = "FedGraphNN:" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.info(args)

    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedmolecule",
            name="(ego) FedGraphNN(d)"
            + str(args.model)
            + "r"
            + str(args.dataset)
            + "-lr"
            + str(args.lr),
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    set_seed(2021)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(
        process_id, worker_number - 1, args.gpu_num_per_server
    )
    logging.info("device = {}".format(device))

    # load data
    dataset, num_cats, feat_dim = load_data(args, args.dataset)
    [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ] = dataset

    logging.info("Dataset Processed")

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    #model, trainer = create_model(args, args.model, feat_dim, num_cats, output_dim=None)

    # start "federated averaging (FedAvg)"
    #fl_alg = get_fl_algorithm_initializer(args.fl_algorithm)
    #fl_alg(
    #    process_id,
    #    worker_number,
    #    device,
    #    comm,
    #    model,
    #    train_data_num,
    #    train_data_global,
    #    test_data_global,
    #    data_local_num_dict,
    #    train_data_local_dict,
    #    test_data_local_dict,
    #    args,
    #    trainer,
    #)
