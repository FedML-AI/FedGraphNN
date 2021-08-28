#!/bin/bash
#set -x


# Declare an array of string with type
declare -a dataset_hps=("citeseer")
declare -a model_hps=("gcn" "sgc" "sage")
declare -a partition_alpha_hps=("0.1")
declare -a round_num_hps=("100")
declare -a epoch_hps=("1" "3" "5")
declare -a batch_size_hpo=("1")
declare -a lr_hpo=("0.01" "0.001")

declare -a hidden_dim_hpo=("128")
declare -a n_layers_hpo=("3")
declare -a dropout_hpo=("0.5")
declare -a weight_decay_hpo=("0.00001")

# Iterate the string array using for loop
# shellcheck disable=SC2068
for dataset in ${dataset_hps[@]}; do
  for model in ${model_hps[@]}; do
    for partition_alpha in ${partition_alpha_hps[@]}; do
      for round_num in ${round_num_hps[@]}; do
        for epoch in ${epoch_hps[@]}; do
          for batch_size in ${batch_size_hpo[@]}; do
            for lr in ${lr_hpo[@]}; do
              for hidden_dim in ${hidden_dim_hpo[@]}; do
                for n_layers in ${n_layers_hpo[@]}; do
                  for dropout in ${dropout_hpo[@]}; do
                    for weight_decay in ${weight_decay_hpo[@]}; do
                      echo $dataset + $model + $partition_alpha + $round_num + $epoch + $batch_size + $lr + $hidden_dim \
                       + $n_layers + $dropout + $weight_decay
                       sh run_fed_node_clf.sh 10 10 1 8 $model hetero $partition_alpha $round_num \
                      $epoch $batch_size $lr $hidden_dim $n_layers $dropout $weight_decay $dataset
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
#
#
