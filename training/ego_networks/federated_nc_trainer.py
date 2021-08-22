import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score, confusion_matrix

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC

class FedNodeClfTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        val_data, test_data = None, None
        try:
            val_data = self.val_data
            test_data = self.test_data
        except:
            pass

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.wd)
        else:
            optimizer = torch.optim.Adam( model.parameters(), lr=args.lr,  weight_decay = args.wd)


        max_test_score, max_val_score = 0 , 0
        best_model_params = {}
        for epoch in range(args.epochs):
            nnodes = 0
            acc_sum = 0

            for idx_batch, batch in enumerate(train_data):
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.y
                acc_sum += pred.argmax(dim=1).eq(label).sum().item()
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                nnodes += label.size(0)
            acc = acc_sum / nnodes

            if val_data is not None:
                    val_score, _ = self.test(self.val_data, device)
                    print('Epoch = {}, Iter = {}/{}: Val F1 = {}'.format(epoch, idx_batch + 1, len(train_data), acc))
                    if val_score > max_val_score:
                        max_test_score = val_score
                        best_model_params = {k: v.cpu() for k, v in model.state_dict().items()}
                    print('Current best validation = {}'.format(val_score))

            if ((idx_batch + 1) % args.frequency_of_the_test == 0) or (idx_batch == len(train_data) - 1):
                if test_data is not None:
                    test_score, _ = self.test(self.test_data, device)
                    print('Epoch = {}, Iter = {}/{}: Test F1 = {}'.format(epoch, idx_batch + 1, len(train_data), acc))
                    if test_score > max_test_score:
                        max_test_score = test_score
                        best_model_params = {k: v.cpu() for k, v in model.state_dict().items()}
                    print('Current best = {}'.format(max_test_score))

        return max_test_score, best_model_params

    def test(self, test_data, device):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)
        conf_mat = np.zeros((self.model.nclass, self.model.nclass))
        

        with torch.no_grad():
            for batch in test_data:
                batch.to(device)
                
                pred = model(batch)
                label = batch.y
                conf_mat+= confusion_matrix(label.cpu().numpy() , pred.argmax(dim=1).cpu().numpy(), labels = np.arange(0,self.model.nclass) )

        #Compure Micro F1   
        TP = np.trace(conf_mat)
        FP = np.sum(conf_mat, sim = -1) - TP
        FN = FP
        micro_pr = TP / (TP + FP)
        micro_rec = TP /( TP + FN)
        micro_F1 = 2* micro_pr * micro_rec / (micro_pr + micro_rec)
        #Compute Macro-F1
        macro_F1s = np.zeros((1, self.model.nclass))
        for i in range(self.model.nclass):
            pr = conf_mat[i,i] /(np.sum(conf_mat[:,i]) - conf_mat[i,i])
            rec = conf_mat[i,i] /(np.sum(conf_mat[i,:]) - conf_mat[i,i])
            macro_F1s[i] = 2 * pr * rec / (pr  +rec)

        macro_F1 = np.mean(macro_F1s)
        # score = f1_score(truth, preds, labels  = label_set, average = 'micro')
        # print(score)
        score = (micro_F1 ,  macro_F1)
        return score, model

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, micro_list, macro_list = [], [] , []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            micro_list.append(score[0])
            macro_list.append(score[1])
            logging.info('Client {}, Test Micro F1 = {}'.format(client_idx, score[0]))
            logging.info('Client {}, Test Macro F1 = {}'.format(client_idx, score[1]))
            wandb.log({"Client {} Test/Micro F1".format(client_idx): score[0]})
            wandb.log({"Client {} Test/Macro F1".format(client_idx): score[1]})

        avg_micro = np.mean(np.array(micro_list))
        avg_macro = np.mean(np.array(macro_list))
        logging.info('Test Micro F1 = {} , Macro F1 = {}'.format(avg_micro , avg_macro))
        wandb.log({"Test/ Micro F1": avg_micro})
        wandb.log({"Test/ Macro F1": avg_macro})

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info('Models match perfectly! :)')
