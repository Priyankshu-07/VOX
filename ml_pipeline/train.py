import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import IntentDataset
from tokenizer import WordTokenizer
from model import BiGRUIntentModel
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_model():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    model_dir = os.path.join(os.path.dirname(__file__), '../model')
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = WordTokenizer(max_seq_length=20)  
    print("Loading datasets...")
    train_dataset = IntentDataset(os.path.join(data_dir, 'train.csv'), tokenizer, is_train=True)
    val_dataset = IntentDataset(os.path.join(data_dir, 'val.csv'), tokenizer, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    tokenizer.save(os.path.join(model_dir, 'vocab.json'))
    print(f"Tokenizer saved. Vocab size: {tokenizer.vocab_size}")
    model = BiGRUIntentModel(vocab_size=tokenizer.vocab_size, embed_dim=64, hidden_dim=64, num_classes=5)
    print(f"Model Parameters: {count_parameters(model):,} (Target < 1,000,000)")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    epochs = 10
    best_val_loss = float('inf')
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch['input_ids'], batch['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['input_ids'], batch['label']
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print("  --> Saved Best Model")         
    print("\nTraining Complete!")
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    model.eval()
    print("Quantizing model for CPU inference (INT8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), os.path.join(model_dir, 'quantized_model.pth'))
    print(f"Quantized Model Size: {os.path.getsize(os.path.join(model_dir, 'quantized_model.pth')) / 1024:.2f} KB")
    print("Exporting model to ONNX...")
    dummy_input = torch.zeros(1, tokenizer.max_seq_length, dtype=torch.long)
    onnx_path = os.path.join(model_dir, 'model.onnx')
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input_ids'], 
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")
if __name__ == '__main__':
    train_model()
