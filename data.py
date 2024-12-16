"""
This is where we load and preprocess the data
"""
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import json
from typing import List, Dict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, Optional

class OrderedNQDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features = self._load_and_process_data(data_path)
        
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """Load and process all examples from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)
            # data = data[:50]
            
        features = []
        not_equal = 0
        for item in data:
            question = item['questions'][0]['input_text']
            #not sure if this will ever happen
            if len(item['questions']) > 1:
                print(len(item['questions']))
            context = item['contexts']
            
            # Initialize positions
            start_position = 0
            end_position = 0
            answer_type = 0  # Default to no_answer (0)
            answer_text = ""  # Initialize answer_text
            
            
            if len(item.get('answers', [])) > 0:
                answer = item['answers'][0]
                answer_type = 1 if answer['input_text'] == 'short' else 0  # 1 for short, 0 for no_answer
                
                if answer['input_text'] == 'short':
                    # Get the answer text and tokenize
                    question_tokens = self.tokenizer.tokenize(question)
                    answer_text = context[answer['span_start']:answer['span_end']]
                    
                    # Get context before answer and tokenize
                    context_before = context[:answer['span_start']]
                    context_before_tokens = self.tokenizer.tokenize(context_before)
                    
                    # Get answer tokens
                    answer_tokens = self.tokenizer.tokenize(answer_text)
                    
                    # Calculate positions including special tokens
                    # [CLS] question [SEP] context [SEP]
                    start_position = 1 + len(question_tokens) + 1 + len(context_before_tokens)
                    end_position = start_position + len(answer_tokens)
                    
                    # If start position or end position exceeds max sequence length, treat as no_answer
                    if start_position >= self.max_seq_length or end_position >= self.max_seq_length:
                        start_position = 0
                        end_position = 0
                        answer_type = 0
            
            # Encode full sequence
            encoded = self.tokenizer(
                text=question,
                text_pair=context,
                padding='max_length',
                truncation='only_second',
                max_length=self.max_seq_length,
                return_tensors=None
            )

            # Only verify answer span if we have an answer
            if answer_text:
                predicted_answer = self.tokenizer.decode(
                    encoded['input_ids'][start_position:end_position],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                actual_answer = self.tokenizer.decode(
                    self.tokenizer(text=answer_text)['input_ids'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                if answer_text:
                    # Normalize both predicted and actual answers
                    predicted_answer = predicted_answer.lower().strip()
                    actual_answer = actual_answer.lower().strip()
                    # Check if either is contained in the other
                    if (predicted_answer not in actual_answer) and (actual_answer not in predicted_answer):
                        not_equal +=1
                        continue
            
            feature = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'start_position': start_position,
                'end_position': end_position,
                'answer_type': answer_type,
            }
            
            # Final validation of positions
            if start_position >= self.max_seq_length or end_position > self.max_seq_length:
                feature['start_position'] = 0
                feature['end_position'] = 0
                feature['answer_type'] = 0
            
            features.append(feature)
            
        
        return features
    def __len__(self) -> int:
            return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single instance."""
        feature = self.features[idx]
        
        return {
            'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
            'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
            'answer_type': torch.tensor(feature['answer_type'], dtype=torch.long),
        }
        
@dataclass
class DataInfo:
    """Container for dataloader and sampler."""
    dataloader: DataLoader
    sampler: Optional[DistributedSampler] = None

def get_qa_dataset(
    args,
    tokenizer,
    is_training: bool,
):
    """Create dataset and dataloader for Natural Questions."""
    # Select appropriate data path based on training/eval
    data_path = args.train_data_path if is_training else args.eval_data_path
    
    # Create dataset
    dataset = OrderedNQDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Setup sampler for distributed training
    # sampler = DistributedSampler(dataset) if args.distributed and is_training else None
    sampler = None
    shuffle = is_training
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        generator=g,
        drop_last=is_training
    )
    
    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_data(args, tokenizer) -> Dict[str, DataInfo]:
    """Get train and eval dataloaders."""
    data = {}
    
    if hasattr(args, 'train_data_path'):
        data["train"] = get_qa_dataset(
            args=args,
            tokenizer=tokenizer,
            is_training=True
        )
    
    if hasattr(args, 'eval_data_path'):
        data["eval"] = get_qa_dataset(
            args=args,
            tokenizer=tokenizer,
            is_training=False
        )
    
    return data