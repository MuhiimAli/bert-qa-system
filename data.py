import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import json
from typing import List, Dict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, Optional

class OrderedNQDataset(Dataset):
    """Dataset class for Natural Questions with ordered short/no-answer examples."""
    
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
            # if len(data) > 50:
            #     print(f"Limiting dataset from {len(data)} to 50 examples for debugging")
            #     data = data[:50]
            
        features = []
        for item in data:
            # Process each example
            question = item['questions'][0]['input_text']  # Take first question
            context = item['contexts']
            
            # Encode question and context together using encode_plus
            encoded = self.tokenizer.encode_plus(
                question,
                context,
                padding='max_length',
                add_special_tokens=True,
                truncation='only_second',
                max_length=self.max_seq_length,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            # Squeeze tensors to remove batch dimension
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            token_type_ids = encoded['token_type_ids'].squeeze(0)
            
            # Get the positions of special tokens
            input_ids_list = input_ids.tolist()
            sep_positions = [i for i, token_id in enumerate(input_ids_list) if token_id == self.tokenizer.sep_token_id]
            
            # Verify the structure: [CLS] question [SEP] context [SEP]
            if len(sep_positions) < 2:
                print(f"Warning: Example {item['id']} doesn't have expected token structure")
                continue
                
            question_end = sep_positions[0]  # Position of first [SEP]
            
            if len(item.get('answers', [])) > 0:
                answer = item['answers'][0]
                answer_type = answer['input_text']
                
                if answer_type == 'short':
                    # Adjust answer spans to account for question tokens and special tokens
                    context_start = question_end + 1
                    start_position = context_start + answer['span_start']
                    end_position = context_start + answer['span_end']
                    
                    # Ensure positions are within bounds
                    if start_position >= self.max_seq_length:
                        start_position = 0
                        end_position = 0
                        answer_type = 'no-answer'
                    else:
                        end_position = min(end_position, self.max_seq_length - 1)
                else:
                    start_position = 0
                    end_position = 0
                    answer_type = 'no-answer'
            else:
                start_position = 0
                end_position = 0
                answer_type = 'no-answer'
            
            feature = {
                'input_ids': input_ids.tolist(),
                'attention_mask': attention_mask.tolist(),
                'token_type_ids': token_type_ids.tolist(),
                'start_position': start_position,
                'end_position': end_position,
                'answer_type': answer_type,
                'example_id': item['id']
            }
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
            'token_type_ids': torch.tensor(feature['token_type_ids'], dtype=torch.long),
            'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
            'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
            'answer_type': feature['answer_type'],
            'example_id': feature['example_id']
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
    
    # Setup sampler for distributed training
    # sampler = DistributedSampler(dataset) if args.distributed and is_training else None
    sampler = None
    shuffle = is_training and sampler is None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
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