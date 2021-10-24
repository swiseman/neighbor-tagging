Updated code for [Label-Agnostic Sequence Labeling by Copying Nearest Neighbors](https://www.aclweb.org/anthology/P19-1533.pdf). These updates include:
- using (recent) Huggingface libraries where possible
- dropping out neighbor embeddings when training
which lead to better performance.


# Dependencies
- transformers (tested with version 4.10.2)
- datasets (tested with version 1.12.1)
- seqeval (tested with version 1.2.2)


# Training
To train the neighbor-based NER model, run:
```
python -u train_words.py -cuda -db_fi ner.db -drop_db -detach_db -save mynermodel.pt
```

By default the above script will save a database file to the argument of `-db_fi`. Once the database file has been saved, you can rerun the above with `-load_saved_db` to avoid retokenizing everything.

# Evaluation
To evaluate on the development set, run:
```
python -u train_words.py -cuda -db_fi ner.db -load_saved_db -train_from mynermodel.pt -just_eval "test"
```

To run evaluation with recomputed neighbors (under the fine-tuned, rather than pretrained BERT), use the option `-just_eval test-newne`.

The `-dp_pred` and `-c` options can be used for DP decoding.
