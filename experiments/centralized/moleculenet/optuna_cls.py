"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.
"""
import argparse
import os
import random
import sys

import torch
import numpy as np
import torch.nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from data_preprocessing.molecule.data_loader import get_dataloader, get_data
from model.sage_readout import SageMoleculeNet
from model.gat_readout import GatMoleculeNet
from model.gcn_readout import GcnMoleculeNet
from training.sage_readout_trainer import SageMoleculeNetTrainer
from training.gat_readout_trainer import GatMoleculeNetTrainer
from training.gcn_readout_trainer import GcnMoleculeNetTrainer

import optuna
from optuna.samplers import TPESampler


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--dataset', type=str, default='sider', help='Dataset used for training')

    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')

    parser.add_argument('--normalize_features', type=bool, default=False,
                        help='Whether or not to symmetrically normalize feat matrices')

    parser.add_argument('--normalize_adjacency', type=bool, default=False,
                        help='Whether or not to symmetrically normalize adj matrices')

    parser.add_argument('--sparse_adjacency', type=bool, default=False,
                        help='Whether or not the adj matrix is to be processed as a sparse matrix')

    parser.add_argument('--model', type=str, default='graphsage', help='Model name. Currently supports SAGE, GAT and GCN.')

    parser.add_argument('--hidden_size', type=int, default=32, help='Size of GNN hidden layer')

    parser.add_argument('--node_embedding_dim', type=int, default=32,
                        help='Dimensionality of the vector space the atoms will be embedded in')

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for LeakyRelu used in GAT')

    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads used in GAT')

    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout used between GraphSAGE layers')

    parser.add_argument('--readout_hidden_dim', type=int, default=64, help='Size of the readout hidden layer')

    parser.add_argument('--graph_embedding_dim', type=int, default=64,
                        help='Dimensionality of the vector space the molecule will be embedded in')

    parser.add_argument('--client_optimizer', type=str, default='adam', metavar="O",
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                        help='learning rate (default: 0.0015)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--frequency_of_the_test', type=int, default=5, help='How frequently to run eval')

    parser.add_argument('--device', type=str, default="cuda:0", metavar="DV", help='gpu device for training')

    args = parser.parse_args()

    return args


def update_args_(args, params):
    """updates args in-place"""
    dargs = vars(args)
    dargs.update(params)


def main(trial=None):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if trial is not None:
        params = {
                  'dropout': trial.suggest_float("dropout", 0.3, 0.6), \
                  'lr': trial.suggest_float("lr", 1e-3, 1e-2, log=True), \
                  }
        update_args_(args, params)
    dataset_path = args.data_dir + "/" + args.dataset
    compact = (args.model == 'graphsage')

    if (args.dataset != 'sider') and (args.dataset != 'clintox') and (args.dataset != 'bbbp') and \
            (args.dataset != 'bace') and (args.dataset != 'hiv') and (args.dataset != 'muv') and \
            (args.dataset != 'tox21') and (args.dataset != 'toxcast'):
        raise Exception("no such dataset!")

    train_data, val_data, test_data = get_dataloader(dataset_path, compact=compact,
                                                     normalize_features=args.normalize_features,
                                                     normalize_adj=args.normalize_adjacency)
    _, feature_matrices, labels = get_data(dataset_path)

    feat_dim = feature_matrices[0].shape[1]
    num_cats = labels[0].shape[0]
    print("feat_dim = %d" % feat_dim)
    print("num_cats = %d" % num_cats)
    del feature_matrices
    del labels

    if args.model == 'graphsage':
        model = SageMoleculeNet(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                args.readout_hidden_dim, args.graph_embedding_dim, num_cats)
        trainer = SageMoleculeNetTrainer(model)
    elif args.model == 'gat':
        params = {
            'num_heads': trial.suggest_int("num_heads", 1, 7),
            'alpha': trial.suggest_categorical("alpha", [2, 4])
        }
        update_args_(args, params)
        model = GatMoleculeNet(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                               args.alpha, args.num_heads, args.readout_hidden_dim, args.graph_embedding_dim, num_cats)
        trainer = GatMoleculeNetTrainer(model)
    elif args.model == 'gcn':
        model = GcnMoleculeNet(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                               args.readout_hidden_dim, args.graph_embedding_dim, num_cats,
                               sparse_adj=args.sparse_adjacency)
        trainer = GcnMoleculeNetTrainer(model)
    else:
        raise Exception('No such model')

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device == 'cuda:0') else "cpu")

    trainer.test_data = test_data
    max_test_score, best_model_params = trainer.train(train_data, device, args)

    return max_test_score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")

    study.optimize(main, n_trials=25)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
