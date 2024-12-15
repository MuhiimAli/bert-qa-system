# bert-qa
# Project Overview
 Our project focuses on fine-tuning BERT for a Question Answering (QA) task. The model takes a question and a context as input and predicts the token span of the answer within the context. If no answer is present, it predicts a span pointing to the [CLS] token. We followed the paper "A BERT Baseline for the Natural Questions", implementing the loss function from Section 3 and data preprocessing techniques from Section 2. While the paper covers four answer types, our dataset is simplified to two: no answer and short answer.
We evaluate the model using three metrics: precision, recall, and F1 score. Our goal is to replicate the Short Answer Dev performances reported in the paper.

## How To Run
First, install the required packages:

```bash
pip install -r requirements.txt
```


```bash
python main.py \
    --train_data_path /path/to/train.json \
    --eval_data_path /path/to/eval.json \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --num_epochs 2 \
    --weight_decay 0.01 \
    --num_workers 4 \
```
