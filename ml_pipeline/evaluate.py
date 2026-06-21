import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from dataset import IntentDataset, REV_INTENT_MAP
from tokenizer import WordTokenizer
from model import BiGRUIntentModel

def evaluate_model():
    model_dir = os.path.join(os.path.dirname(__file__), '../model')
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    print("Loading tokenizer & vocabulary...")
    tokenizer = WordTokenizer()
    tokenizer.load(os.path.join(model_dir, 'vocab.json'))
    
    print("Loading best PyTorch model...")
    model = BiGRUIntentModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    model.eval()

    test_dataset = IntentDataset(os.path.join(data_dir, 'test.csv'), tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'])
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(batch['label'].numpy())
    target_names = [REV_INTENT_MAP[i] for i in range(len(REV_INTENT_MAP))]
    print("\n--- Final Test Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")

if __name__ == '__main__':
    evaluate_model()
