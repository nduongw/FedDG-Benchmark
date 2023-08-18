# domain-generalization-fed-learning

This is the git repo of [Benchmarking Algorithms for Domain Generalization in Federated Learning]([https://openreview.net/forum?id=IsCg7qoy8i9](https://openreview.net/forum?id=EqGjKubKEB)).

## Available methods
* FedAvg
* IRM
* REx
* Fish
* MMD
* DeepCoral
* GroupDRO
* FedProx
* Scaffold
* FedDG
* FedADG
* FedSR
* FedGMA

# Environment preparation
```
conda create  --name <env> --file requirements.txt
```

## Prepare Datasets
All datasets derived from [Wilds](https://wilds.stanford.edu/) Datasets. We also implement [femnist](https://leaf.cmu.edu/) and [PACS](https://arxiv.org/abs/2007.01434) datasets.

### Preparing metadata.csv and RELEASE_v1.0.txt
For PACS and FEMNIST dataset, please put 
```
resources/femnist_v1.0/* 
```
and 
```
resources/pacs_v1.0/* 
```
into your dataset directory.

### Preparing fourier transformation
Some methods require fourier transformation. To accelerate training, we should prepare the transformation data in advance. Please first load the scripts in the scripts path. Note: Please config the root_path in the script.

## Run Experiments
To run the experiments, simply prepare your config file $config_path, and run
```
python main.py --config_file $config_path
```
For example, to run fedavg-erm with centralized learning on iwildcam, run
```
python main.py --config_file ./config/ERM/iwildcam/centralized.json
```

## Configuration Explaination
The config file is json format containing four keys: "global", "server", "client", "dataset". Here's a demo
```
{
    "global": {
        "log_path": "./log",                              # log file.
        "seed": 8989,                                     # random seed
        "num_clients": 100,                               # number of clients/shards
        "dataset_name": "CivilComments",                  # dataset name.
        "id": 0,                                          # experiment identifier.
        "batch_size": 16                                  # batch_size
    },
    "server": {
        "mp": false,                                      # multiprocessing or not.
        "algorithm": "FedAvg",                            # server side method.
        "fraction": 1,                                    # The participation ratio of the clients each round.
        "num_rounds": 5                                   # times of communication/aggregation.
    },
    "client": {
        "algorithm": "Coral",                             # client side method.
        "local_epochs": 1,                                # epochs of running local datasets between each communication/aggregation. 
        "n_groups_per_batch": 2,                          # number of domains contained in each batch.
        "optimizer": "torch.optim.Adam",                  # client side optimization algorithm.
        "optimizer_config": {                             # config of optimizer
            "lr": 1e-05
        },
        "penalty_weight": 1                               # special hyperparameters for each method.
    },
    "dataset": {
        "data_path": "/local/scratch/a/shared/datasets/", # datasets direction location.
        "iid": 1                                          # \lambda. 1: domain homogeneity. 0: domain seperation.
    }
}
```
