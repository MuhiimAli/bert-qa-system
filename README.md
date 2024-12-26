# Bert-QA
# Project Overview
 Our project focuses on fine-tuning BERT for a Question Answering (QA) task. The model takes a question and a context as input and predicts the token span of the answer within the context. If no answer is present, it predicts a span pointing to the [CLS] token. We followed the paper "A BERT Baseline for the Natural Questions", implementing the loss function from Section 3 and data preprocessing techniques from Section 2. While the paper covers four answer types, our dataset is simplified to two: no answer and short answer.
We evaluate the model using three metrics: precision, recall, and F1 score. Our goal is to replicate the Short Answer Dev performances reported in the paper.

## Set up
First, install the required packages:

```bash
pip install -r requirements.txt
```

## How to run
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
## Project Structure

* **main.py:** Entry point for the project. Contains the main function that can be run either through VSCode's "Run Python File" or through the terminal with custom arguments.

* **data.py:** Handles dataset loading and preprocessing according to the paper's methodology.

* **train.py:** Manages training and evaluation loops. Features:
  - Metric computation during evaluation
  - train_one_epoch method for training
  - evaluate method for testing
  - Validation metrics calculation and reporting after each epoch

* **loss.py:** Custom implementation of the loss function using manual cross-entropy loss calculation as recommended by TAs.


## Model Architecture

The model is built on top of DistilBERT and has been adapted for question answering tasks. It uses three separate linear layers, each responsible for a different aspect of answer prediction:

1. Start Position: Linear layer to predict where the answer starts in the text
2. End Position: Linear layer to predict where the answer ends in the text
3. Answer Type: Linear layer to classify the type of answer

The model takes a question-answer pair as input, processes it through DistilBERT, and then uses these three linear layers to make predictions. 

## Hyperparameters

Best performing configuration:
- Learning rate: 0.00003882259533696199
- Number of epochs: 2
- Number of workers: 4
- Random seed: 42
- Weight decay: 0

## Model Performance

Final metrics achieved:
- Precision: 0.6131
- Recall: 0.4786
- F1: 0.5376

For detailed training logs and complete output, refer to the output.log and output.log1 files generated during training.
