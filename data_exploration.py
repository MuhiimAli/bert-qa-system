"""
Data analysis functionality for Natural Questions dataset
"""
from data import OrderedNQDataset
from transformers import DistilBertTokenizer
import argparse
from collections import Counter

class DataAnalysis:
    def __init__(self, dataset: OrderedNQDataset):
        self.dataset = dataset
        
    def analyze_answer_lengths(self):
        """Analyze answer lengths in the dataset."""
        answer_lengths = []
        answer_types = Counter()
        
        for feature in self.dataset.features:
            start_pos = feature['start_position']
            end_pos = feature['end_position']
            answer_type = feature['answer_type']
            
            # Count answer types
            answer_types[answer_type] += 1
            
            # Only measure length for actual answers (not no-answer cases)
            if answer_type == 1:  # short answer
                answer_length = end_pos - start_pos
                answer_lengths.append(answer_length)
        
        stats = {
            'total_examples': len(self.dataset),
            'answer_types': dict(answer_types),
            'answer_lengths': {
                'max': max(answer_lengths) if answer_lengths else 0,
                'min': min(answer_lengths) if answer_lengths else 0,
                'avg': sum(answer_lengths)/len(answer_lengths) if answer_lengths else 0
            }
        }
        
        return stats
    
    def print_statistics(self, stats):
        """Print formatted statistics."""
        print("\nDataset Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        
        print("\nAnswer Types:")
        total = stats['total_examples']
        for ans_type, count in stats['answer_types'].items():
            type_name = 'short answer' if ans_type == 1 else 'no answer'
            print(f"{type_name}: {count} ({count/total*100:.1f}%)")
        
        print("\nAnswer Lengths (in tokens):")
        lengths = stats['answer_lengths']
        print(f"Maximum: {lengths['max']}")
        print(f"Minimum: {lengths['min']}")
        print(f"Average: {lengths['avg']:.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', 
                        type=str, 
                        default="/users/mali37/scratch/bert_qa_data/all_train.json",
                        help='Path to training data JSON file')
    parser.add_argument('--eval_data_path', 
                        type=str, 
                        default="/users/mali37/scratch/bert_qa_data/all_dev.json",
                        help='Path to evaluation data JSON file')
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        default=512,
                        help='Maximum sequence length')
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Analyze training data
    print("\nAnalyzing Training Data:")
    train_dataset = OrderedNQDataset(args.train_data_path, tokenizer, args.max_seq_length)
    train_analyzer = DataAnalysis(train_dataset)
    train_stats = train_analyzer.analyze_answer_lengths()
    train_analyzer.print_statistics(train_stats)
    
    # Analyze eval data if provided
    if args.eval_data_path:
        print("\nAnalyzing Evaluation Data:")
        eval_dataset = OrderedNQDataset(args.eval_data_path, tokenizer, args.max_seq_length)
        eval_analyzer = DataAnalysis(eval_dataset)
        eval_stats = eval_analyzer.analyze_answer_lengths()
        eval_analyzer.print_statistics(eval_stats)

if __name__ == '__main__':
    main()