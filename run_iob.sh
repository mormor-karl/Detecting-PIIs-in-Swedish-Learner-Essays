export MAX_LENGTH=512
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=multi-iob-model-weighted-100
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py \
--task_type NER \
--data_dir ../data/ \
--labels ../data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
