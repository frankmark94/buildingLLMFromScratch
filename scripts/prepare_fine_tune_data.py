#!/usr/bin/env python3
"""
Prepare and format datasets for fine-tuning.
Converts various data formats to fine-tuning compatible format.
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def load_text_file(file_path: str, chunk_size: int = 1000) -> List[Dict[str, str]]:
    """Load plain text file and chunk into training examples."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into chunks
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Skip very short chunks
            chunks.append({'text': chunk.strip()})
    
    return chunks


def load_csv_file(file_path: str, text_column: str, 
                  label_column: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load CSV file for fine-tuning."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {'text': row[text_column]}
            if label_column and label_column in row:
                # Convert label to integer if possible
                try:
                    item['label'] = int(row[label_column])
                except ValueError:
                    item['label'] = row[label_column]
            data.append(item)
    
    return data


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If data is not a list, try to extract from common structures
    if not isinstance(data, list):
        if 'data' in data:
            data = data['data']
        elif 'examples' in data:
            data = data['examples']
        elif 'items' in data:
            data = data['items']
        else:
            raise ValueError("JSON file should contain a list or have 'data'/'examples'/'items' key")
    
    return data


def format_for_instruction_tuning(data: List[Dict], instruction_template: str) -> List[Dict[str, str]]:
    """Format data for instruction tuning using a template."""
    formatted_data = []
    
    for item in data:
        # Replace placeholders in template
        text = instruction_template
        for key, value in item.items():
            text = text.replace(f"{{{key}}}", str(value))
        
        formatted_data.append({'text': text})
    
    return formatted_data


def create_conversation_format(data: List[Dict], user_key: str, assistant_key: str) -> List[Dict[str, str]]:
    """Convert conversation data to training format."""
    formatted_data = []
    
    for item in data:
        if user_key in item and assistant_key in item:
            conversation = f"Human: {item[user_key]}\n\nAssistant: {item[assistant_key]}"
            formatted_data.append({'text': conversation})
    
    return formatted_data


def filter_and_clean(data: List[Dict], min_length: int = 10, 
                     max_length: int = 10000, clean_text: bool = True) -> List[Dict]:
    """Filter and clean data."""
    cleaned_data = []
    
    for item in data:
        text = item.get('text', '')
        
        # Length filtering
        if len(text) < min_length or len(text) > max_length:
            continue
        
        # Basic cleaning
        if clean_text:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            # Remove very repetitive content
            words = text.split()
            if len(set(words)) / len(words) < 0.1:  # Too repetitive
                continue
            item['text'] = text
        
        cleaned_data.append(item)
    
    return cleaned_data


def save_data(data: List[Dict], output_path: str, format: str = 'jsonl') -> None:
    """Save processed data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    
    # Input/Output
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=['auto', 'txt', 'csv', 'json', 'jsonl'], 
                       default='auto', help="Input format")
    
    # Data processing
    parser.add_argument("--task-type", choices=['text_generation', 'classification', 'instruction'], 
                       default='text_generation', help="Task type")
    parser.add_argument("--text-column", default='text', help="Text column name")
    parser.add_argument("--label-column", help="Label column name (for classification)")
    
    # Text processing
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Chunk size for plain text files")
    parser.add_argument("--min-length", type=int, default=10, 
                       help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=10000, 
                       help="Maximum text length")
    parser.add_argument("--no-clean", action="store_true", 
                       help="Skip text cleaning")
    
    # Special formats
    parser.add_argument("--instruction-template", 
                       help="Template for instruction tuning (use {key} for placeholders)")
    parser.add_argument("--conversation-user", 
                       help="User key for conversation format")
    parser.add_argument("--conversation-assistant", 
                       help="Assistant key for conversation format")
    
    # Output options
    parser.add_argument("--output-format", choices=['json', 'jsonl'], 
                       default='jsonl', help="Output format")
    parser.add_argument("--sample", type=int, 
                       help="Sample N examples for testing")
    
    args = parser.parse_args()
    
    print("📊 Fine-tuning Data Preparation")
    print("=" * 40)
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 Task: {args.task_type}")
    print("=" * 40)
    
    # Determine input format
    input_path = Path(args.input)
    if args.format == 'auto':
        if input_path.suffix == '.txt':
            format = 'txt'
        elif input_path.suffix == '.csv':
            format = 'csv'
        elif input_path.suffix == '.json':
            format = 'json'
        elif input_path.suffix == '.jsonl':
            format = 'jsonl'
        else:
            raise ValueError(f"Cannot auto-detect format for {input_path.suffix}")
    else:
        format = args.format
    
    print(f"📄 Detected format: {format}")
    
    # Load data
    try:
        if format == 'txt':
            data = load_text_file(args.input, args.chunk_size)
        elif format == 'csv':
            data = load_csv_file(args.input, args.text_column, args.label_column)
        elif format == 'json':
            data = load_json_file(args.input)
        elif format == 'jsonl':
            data = load_jsonl_file(args.input)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Loaded {len(data)} examples")
    
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Apply transformations
    if args.instruction_template:
        print("🔄 Applying instruction template")
        data = format_for_instruction_tuning(data, args.instruction_template)
    
    elif args.conversation_user and args.conversation_assistant:
        print("🔄 Converting to conversation format")
        data = create_conversation_format(data, args.conversation_user, args.conversation_assistant)
    
    # Filter and clean
    print("🧹 Filtering and cleaning data")
    original_count = len(data)
    data = filter_and_clean(
        data, 
        min_length=args.min_length,
        max_length=args.max_length,
        clean_text=not args.no_clean
    )
    print(f"📊 Kept {len(data)}/{original_count} examples after filtering")
    
    # Sample if requested
    if args.sample and len(data) > args.sample:
        import random
        data = random.sample(data, args.sample)
        print(f"🎲 Sampled {len(data)} examples")
    
    # Save data
    try:
        save_data(data, args.output, args.output_format)
        print(f"✅ Data saved to {args.output}")
        
        # Show examples
        print("\n📝 Example data:")
        for i, item in enumerate(data[:3]):
            print(f"\n[{i+1}]")
            if 'text' in item:
                text = item['text'][:200] + "..." if len(item['text']) > 200 else item['text']
                print(f"Text: {text}")
            if 'label' in item:
                print(f"Label: {item['label']}")
        
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())