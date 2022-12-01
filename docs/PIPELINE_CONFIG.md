# Pipeline Configuration File

Here we provide documents for all arguments in the pipeline configuration file, where their corresponding implementation can be found in `main.py`.

## System-related Arguments
```yaml
SYSTEM:
  NUM_GPUS: 4
  MAX_MEMORY_IN_GB: 16
```

- `NUM_GPUS`: the number of available GPUs in the GPU server. GPU 0, 1, ..., NUM_GPUS-1 will be used for the pipeline. Currently we support NUM_GPUS=1/2/4/8.
- `MAX_MEMORY_IN_GB`: (optional) the maximum available RAM in the server, e.g. 16 GB in the example. This will trigger the memory-time tradeoff mechanism during data preprocessing and assures that data preprocessing will not use over 16 GB RAM.

## Model-related Arguments:
```yaml
TOKENIZER:
  NAME_OR_PATH: bert-large-uncased
```
- `TOKENIZER.NAME_OR_PATH`: tokenizer used by dataset preprocessing, pretraining and finetuning. It will check if it is the name of a [huggingface model](https://huggingface.co/models). If so, the corresponding tokenizer will be used. Otherwise, it will be assumed to be the local path to tokenizer files in huggingface format.

## Dataset-related Arguments
```yaml
DATASET:
  ENABLED: True
  ID: bert-dataset

  HUGGINGFACE_DATASETS:
    - [wikipedia, 20220301.en]
    - [bookcorpusopen, plain_text]

  CUSTOMIZED_DATASETS:
    - /home/data/pile
    - /home/data/customized-dataset
```
- `ENABLED`: (optional) whether this stage will be executed. True by default.
- `ID`: (optional) the dataset id that will be used by the pipeline to create intermediate folders. A unique ID will be automatically generated based on dataset content if not specified by user.
- `HUGGINGFACE_DATASETS`: a list of [huggingface datasets](https://huggingface.co/datasets). Each item should have two values, the dataset name (e.g. wikipedia) and the subset name (e.g. plain_text).
- `CUSTOMIZED_DATASETS`: a list of folder paths that contains customized text datasets. For details of the customized dataset format, please refer to the dedicated [README](../dataset/README.md). Relative paths are supported, where the cloned project directory will server as `.`, i.e. the location where `main.py` is placed.

## Pretrain-related Arguments
```yaml
PRETRAIN:
  ENABLED: True
  NUM_STEPS: 23000
  OPTIMIZER: adamw
  LEARNING_RATE: 1e-3
```
- `ENABLED`: (optional) whether this stage will be executed. True by default.
- `NUM_STEPS`: number of steps that will be used in the pretraining process. 23000 by default. In GeForce 3090 x4 GPUs, 23000 steps correspond to roughly 1.3 days.
- `OPTIMIZER`: (optional) the optimization algorithm for pretraining. [Adamw](https://arxiv.org/abs/1711.05101) by default. Other supported optimizers: SGD, SGD with momentum. Their detailed settings can be found in [conf/optimizer_\*](../conf).
- `LEARNING_RATE`: (optional) the peak learning rate. 1e-3 by default.

## Finetune-related Arguments
```yaml
FINETUNE:
  ENABLED: True
  MULTIPROCESS_GPU_LIST:
    - [0, 1]
    - [2, 3]
```
- `ENABLED`: (optional) whether this stage will be executed. True by default.
- `MULTIPROCESS_GPU_LIST`: (optional) gpu allocation scheme if we use multi-process during finetuning. As hyperparameter search is conducted during finetuning, the whole cost can be very expensive. To speedup finetuning, we adopt multiprocessing, where each process will use 1/2/4/8/16 gpus, so that the hyperparameter search can be conducted in parallel.

## Result-collection-related Arguments
```yaml
RESULT_COLLECTION:
  ENABLED: True
```
- `ENABLED`: (optional) whether this stage will be executed. True by default.

## Other Arguments

### Wandb-related Arguments
```yaml
WANDB:
  API_KEY: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
- `API_KEY`: (optional) the api key provided by [wandb](https://wandb.ai/). We use wandb to track and visualize pretraining/finetuning records, hence users are recommended to use their own wandb accounts and api keys to track records. Otherwise a default key will be used and the record will be send to our wandb account.
