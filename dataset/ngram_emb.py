from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from fasttext import load_model
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help="the name of task")
    parser.add_argument('--ngram_file_path', type=str, required=True, help="the path to a file containing a list of ngrams")
    parser.add_argument('--output_path', type=str, required=True, help="the output ngram embedding file path")
    config = parser.parse_args()

    model = load_model(f"./{config.task_name}.bin")
    ngram_file = open(config.ngram_file_path)
    ngram_emb_path = config.output_path
    all_ngram_emb = []
    for ngram in ngram_file:
        ngram = ngram.strip()
        ngram_list = ngram.split(' ')

        for idx, ngram_word in enumerate(ngram_list):
            ft_word_emb = model.get_word_vector(ngram_word)
            if idx ==0:
                ngram_emb = ft_word_emb
            else:
                ngram_emb += ft_word_emb
        ngram_emb = ngram_emb / len(ngram_list)

        all_ngram_emb.append(ngram_emb)

    all_ngram_emb_numpy = np.array(all_ngram_emb)
    np.save(ngram_emb_path, all_ngram_emb_numpy)
