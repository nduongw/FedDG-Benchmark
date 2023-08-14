# domain-generalization-fed-learning

This is the git repo of [Benchmarking Algorithms for Domain Generalization in Federated Learning]([https://openreview.net/forum?id=IsCg7qoy8i9](https://openreview.net/forum?id=EqGjKubKEB)).

## Environment preparation
```
conda create  --name <env> --file requirements.txt
```

## Prepare Datasets
For PACS and FEMNIST dataset, please put 
```
resources/femnist_v1.0/* 
```
and 
```
resources/pacs_v1.0/* 
```
into your dataset directory.

## Run Experiments
To run the experiments, simply prepare your config file $config_path, and run
```
python main.py --config_file $config_path
```
For example, to run fedavg-erm with centralized learning on iwildcam, run
```
python main.py --config_file ./config/ERM/iwildcam/centralized.json
```

