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
            
        features = []
        for item in data:
            # Process each example
            question = item['questions'][0]['input_text']  # Take first question
            context = item['contexts']
            
            # Tokenize question and context
            question_tokens = self.tokenizer.tokenize(question)
            context_tokens = self.tokenizer.tokenize(context)
            
            # Truncate context if needed
            max_context_length = self.max_seq_length - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
            context_tokens = context_tokens[:max_context_length]
            
            # Combine tokens
            tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Create attention mask and token type IDs
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)
            
            # Pad if necessary
            padding_length = self.max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                token_type_ids += [0] * padding_length
            
            # Determine answer type and positions
            has_answer = len(item.get('answers', [])) > 0
            if has_answer:
                answer = item['answers'][0]
                # Get token-based start and end positions
                # Note: This is simplified - you'd need proper character to token mapping
                offset = len(question_tokens) + 2  # Account for [CLS], question, and [SEP]
                start_position = offset + answer['span_start']  # Simplified
                end_position = offset + answer['span_end']    # Simplified
                answer_type = 'short'
            else:
                # No answer - point to [CLS] token
                start_position = 0
                end_position = 0
                answer_type = 'no-answer'
            
            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
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