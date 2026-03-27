import argparse
import os
import textwrap
import torch
from tokenizers import Tokenizer
from model import Transformer 


def parse_args():
    parser = argparse.ArgumentParser(description="Инференс модели Transformer для суммаризации текстов.")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Путь к входному .txt файлу с исходным текстом.")
    parser.add_argument("--output", "-o", type=str, default="summary.txt", 
                        help="Путь для сохранения сгенерированной аннотации (по умолчанию: summary.txt).")
    parser.add_argument("--weights", "-w", type=str, default="transformer.pth", 
                        help="Путь к весам модели (.pth).")
    parser.add_argument("--tokenizer", "-t", type=str, default="transformer_tokenizer.json", 
                        help="Путь к файлу токенизатора (.json).")
    return parser.parse_args()

def generate_summary(model: torch.nn.Module, tokenizer: Tokenizer, text: str, device: torch.device) -> str:
    """Функция для инференса."""
    input_ids = tokenizer.encode(text).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        pred_ids = model.generate(
            input_tensor, 
            max_len=75, 
            repetition_penalty=1.25, 
            penalty_window=25
        )

    return tokenizer.decode(pred_ids)

def main():
    args = parse_args()

    # Проверка существования файлов
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file was not found: {args.input}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight file not found: {args.weights}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    with open(args.input, "r", encoding="utf-8") as f:
        source_text = f.read().strip()
        source_text = " ".join(source_text.split())
    
    if not source_text:
        raise ValueError("Input file is empty.")

    print("Loading tokenizer and model weights...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    model = Transformer(
        d_model=512, 
        h=8, 
        enc_num_layers=4, 
        dec_num_layers=4, 
        vocab_size=30000, 
        max_seq_len=5000,
        dropout_p=0.0 
    )
    
    weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    print("Summary generation...")
    summary = generate_summary(model, tokenizer, source_text, device)
    wrapped_summary = textwrap.fill(summary, width=80)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(wrapped_summary)
        
    print(f"Summary saved to file: {args.output}")

if __name__ == "__main__":
    main()