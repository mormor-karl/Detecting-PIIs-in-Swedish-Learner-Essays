export MAX_LENGTH=512
export BERT_MODEL=KB/bert-base-swedish-cased

python3 ../bert/scripts/preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 ../bert/scripts/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
python3 ../bert/scripts/preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt