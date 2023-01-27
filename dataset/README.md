# Dataset Processing

Preparing the dataset includes the following steps:

- Obtain textual data
- Process dataset (wikipedia or bookcorpus) and combine into 1 text file using `process_data.py`
- Divide the data into N shards using `shard_data.py`
- Generate samples for training and testing the model using `generate_samples.py`

## Obtaining textual data

Any textual dataset can be processed and used for training a BERT-like model.

In our experiments we trained models using the English section of Wikipedia and the Toronto Bookcorpus [REF].

Wikipedia dumps can be freely downloaded from https://dumps.wikimedia.org/ and can be processed (removing HTML tags, picture, and non textual data) using [Wikiextractor.py](https://github.com/attardi/wikiextractor).

We are unable to provide a source for Bookcorpus dataset.

## Import and Shard Data 

`shard_data.py` is used to shard multiple text files (processed with the script above) into pre-defined number of shards, and to divide the dataset into train and test sets.

Use `shard_data.py` to import and shard common corpus (e.g. Wikipedia and Bookcorpus) or customized corpus easily. Also `shard_data.py` has supported one-click corpus download and sharding of Wikipedia and Bookcorpus dataset in Huggingface without preparing the data in advance.

IMPORTANT NOTE: the number of shards is affected by the duplication factor used when generating the samples (with masked tokens). This means that if 10 training shards are generated with `shard_data.py` and samples are generated with duplication factor 5, the final number of training shards will be 50.
This approach avoids intra-shard duplications that might overfit the model in each epoch.

IMPORTATN NOTE 2: the performance of the sharding script (we might fix in the future) might be slow if you choose to generate a small amount of shards (from our experiment under 100). If you encounter such situation we recommand to generate 256+ shards and then merging them to fewer using the merging script we provide (`merge_shards.py`). See more info the next section.

See `python shard_data.py -h` for the full list of options.

### **Option 1: Download and Shard Wikipedia and Bookcorpus from Huggingface**

Example for downloading and sharding [Wikipedia](https://huggingface.co/datasets/wikipedia) with subset name "20220301.simple":

```bash
python shard_data.py \
    --output_dir <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64\
    --dataset wikipedia 20220301.simple
```

Example for downloading, concatenation and sharding [Wikipedia](https://huggingface.co/datasets/wikipedia) and [Bookcorpus](https://huggingface.co/datasets/bookcorpusopen):

```bash
python shard_data.py \
    --output_dir <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64 \
    --dataset wikipedia 20220301.en \
    --dataset bookcorpusopen plain_text
```

The output of this code would be `--num_train_shards` train files and `--num_test_shards` test files and an one article per line file.

The `--max_memory` determines the largest RAM the process will take. The process is always faster with a higher `--max_memory`. We recommend to leave some extra space since the RAM usage sometimes would fluctuate.

If you want to change the save path of the cache of data downloaded from Huggingface, add the argument `--huggingface_cache_dir`:

```bash
python shard_data.py \
    --output_dir <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64 \
    --dataset wikipedia 20220301.en \
    --dataset bookcorpusopen plain_text \
    --huggingface_cache_dir <cache_dir>
```

### **Option 2: Shard Local Customized Dataset**

Example for sharding user's own corpus found in the input `--dir` into `256` train shards and `128` test shards, with 10% of the samples held-out for the test set:

```bash
python shard_data.py \
    --output_dir <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64 \
    --dataset custom <dir_path_of_text_files>
```

The supported formats of corpus can be founded in the `custom_data_example` file. The code supports the `txt` file with one article per line, one sentence per line or multiple sentences per line. The most important thing is the double blank lines that split different articles, i.e. "\n\n". Examples are as follows:

```
<article 1>
     
<article 2>
     
<article 3>
     
...
```

or

```
<article 1, sentence 1>
<article 1, sentence 2-3>
<article 1, sentence 4>
    
<article 2, sentence 1>
<article 2, sentence 2>
<article 2, sentence 3-115>

<article 3, sentence 1>
<article 3, sentence 2>
<article 3, sentence 3>
    
...
```

In addition, multiple customized datasets can be concatenated with huggingface datasets and then sharded together:

```bash
python shard_data.py \
    --output_dir <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64 \
    --dataset wikitext wikitext-103-v1 \
    --dataset custom <dir_1_of_text_files> \
    --dataset wikipedia 20220301.simple \
    --dataset wikipedia 20220301.fr \
    --dataset custom <dir_2_of_text_files>
```

### **Option 3: Shard Local Wikipedia or Bookcorpus Dataset**

**Data Processing**

Use `process_data.py` for pre-processing wikipedia/bookcorpus datasets into a single text file.

See `python process_data.py -h` for the full list of options.

An example for pre-processing the English Wikipedia xml dataset:

```bash
python process_data.py -f <path_to_xml> -o <wiki_output_dir> --type wiki
```

An example for pre-processing the Bookcorpus dataset:

```bash
python process_data.py -f <path_to_text_files> -o <bookcorpus_output_dir> --type bookcorpus
```

**Data Sharding**

The remaining part is the same as the customized dataset sharding. Example for sharding Wikipedia corpus found in the input `--dir` into `256` train shards and `128` test shard, with 10% of the samples held-out for the test set:

```bash
python shard_data.py \
    --output <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1 \
    --max_memory 64 \
    --dataset custom <wiki_output_dir> \
    --dataset custom <bookcorpus_output_dir>
```

## Merging Shards (optional)

Merging existing shards into fewer shards (while maintaining 2^N shards, for example 256->128 (2:1 ratio)) can be done with `merge_shards.py` script.

See `python merge_shards.py -h` for the full list of options.

Example for merging randomly 2 shards into 1 shard:

```bash
python merge_shards.py \
    --data <path_to_shards_dir> \
    --output_dir <output_dir> \
    --ratio 2 
```

## Samples Generation

Use `generate_samples.py` for generating samples compatible with dataloaders used in the training script.

IMPORTANT NOTE: the duplication factor chosen will multiply the number of final shards by its factor. For example, 10 shards with duplication factor 5 will generate 50 shards (each shard with different randomly generated (masked) samples).

See `python generate_samples.py -h` for the full list of options.

Example for generating shards with duplication factor 10, lowercasing the tokens, masked LM probability of 15%, max sequence length of 128, tokenizer by provided (Huggingface compatible) model named `bert-large-uncased`, max predictions per sample 20 and 16 parallel processes (for processing faster):

```bash
python generate_samples.py \
    --dir <path_to_shards> \
    -o <output_path> \
    --dup_factor 10 \
    --seed 42 \
    --tokenizer_name bert-large-uncased \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \ 
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 16 \
    --Ngram_path <path_to_Ngram> \
    --Ngram_flag 1
```

## Ngram
if we add the Ngram module to the extreme-bert, To extract n-grams for datasets, please run `pmi_ngram.py` with the following parameters:
`--dataset`: the path of training data file
`--output_dir`: the path of output directory,
We provide two ngram.txt here.

To use fasttext to initilize the embedding for the Ngram, please Training word vectors using fasttext on the raw data following https://fasttext.cc/docs/en/unsupervised-tutorial.html
```bath ./fasttext skipgram -input data/fil9 -output result/fil9```
Then use the `ngram_emb.py` to extract the ngram embeddings from pre-trained models.
