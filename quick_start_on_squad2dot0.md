```
export BERT_BASE_DIR=./bert_base_dir/cased_L-12_H-768_A-12
export SQUAD_DIR=./squad_dir

```

```

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --do_lower_case=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=128 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/ \
  --version_2_with_negative=True

```

Let's  find the prediction result
```
user2@xxx:/tmp/squad_base$ mv nbest_predictions.json ~/bert/squad_dir/nbest_predictions.json
user2@xxx:/tmp/squad_base$ mv predictions.json ~/bert/squad_dir/predictions.json
```

```
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

get:
```
{
  "exact": 71.37202055082962,
  "f1": 74.00009893315637,
  "total": 11873,
  "HasAns_exact": 64.97975708502024,
  "HasAns_f1": 70.24345051170118,
  "HasAns_total": 5928,
  "NoAns_exact": 77.74600504625737,
  "NoAns_f1": 77.74600504625737,
  "NoAns_total": 5945
}
```

```
  "exact": 71.37202055082962,
  "f1": 74.00009893315637,
```

!= {"f1": 90.87081895814865, "exact_match": 84.38978240302744}

???




### Out-of-memory issues

All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of
device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely
to encounter out-of-memory issues if you use the same hyperparameters described
in the paper.

The factors that affect memory usage are:

*   **`max_seq_length`**: The released models were trained with sequence lengths
    up to 512, but you can fine-tune with a shorter max sequence length to save
    substantial memory. This is controlled by the `max_seq_length` flag in our
    example code.

*   **`train_batch_size`**: The memory usage is also directly proportional to
    the batch size.

*   **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model
    requires significantly more memory than `BERT-Base`.

*   **Optimizer**: The default optimizer for BERT is Adam, which requires a lot
    of extra memory to store the `m` and `v` vectors. Switching to a more memory
    efficient optimizer can reduce memory usage, but can also affect the
    results. We have not experimented with other optimizers for fine-tuning.
