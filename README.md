Code for [Label-Agnostic Sequence Labeling by Copying Nearest Neighbors](https://www.aclweb.org/anthology/P19-1533.pdf).

**Update 10/2021:** An updated version of the code, which uses the latest version of the Huggingface transformers and datasets libraries is now in the [update](https://github.com/swiseman/neighbor-tagging/tree/update) branch.

# Dependencies
The code was developed and tested with pytorch-pretrained-bert 0.5.1. (So you may need to do something like `pip install pytorch-pretrained-bert==0.5.1`).

# Data
All the data is in data.tar.gz.

# Trained Models
Trained models can be downloaded from [here](https://drive.google.com/drive/folders/1d_2jF6FM0YSfK8smTOoYjOh78vfOsldG?usp=sharing).

# Training
To train the neighbor-based NER model, run:
```
python -u train_words.py -cuda -db_fi ner-cased-first-b16-n50.pt -detach_db -save mynermodel.pt
```

By default the above script will save a database file to the argument of `-db_fi`. Once the database file has been saved, you can rerun the above with `-load_saved_db` to avoid retokenizing everything.

The other models can be trained analogously, by substituting in the correct data files in `data/`; see the options in `train_words.py`. For POS tasks, the `-acc_eval` flag should be used, and we used a batch size of 20 and a learning rate of 2E-05.

# Evaluation
To evaluate on the development set, run:
```
python -u train_words.py -cuda -db_fi ner-cased-first-b16-n50.pt -load_saved_db -train_from mynermodel.pt -eval_ne_per_sent 100 -pred_shard_size 64 -just_eval dev -val_sent_fi data/conll2003/conll2003-dev.words -val_tag_fi data/conll2003/conll2003-dev.nertags
```

To run evaluation with recomputed neighbors (under the fine-tuned, rather than pretrained BERT), use the option `-just_eval dev-newne`.

You can transfer evaluation as follows:
```
python -u train_words.py -cuda -train_from mynermodel.pt -eval_ne_per_sent 100 -pred_shard_size 64 -align_strat first -just_eval "dev-newne" -zero_shot -sent_fi data/onto/train.words -tag_fi data/onto/train.ner -val_sent_fi data/onto/dev.words -val_tag_fi data/onto/dev.ner
```

The `-dp_pred` and `-c` options can be used for DP decoding.
