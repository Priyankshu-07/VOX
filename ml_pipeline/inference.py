import os
import time
import torch
import torch.nn.functional as F
import onnxruntime as ort
from tokenizer import WordTokenizer
from model import BiGRUIntentModel
from dataset import REV_INTENT_MAP
class IntentInferencer:
    def __init__(self, use_onnx: bool = True):
        self.use_onnx = use_onnx
        model_dir = os.path.join(os.path.dirname(__file__), '../model')
        
        self.tokenizer = WordTokenizer()
        self.tokenizer.load(os.path.join(model_dir, 'vocab.json'))
        
        if self.use_onnx:
            self.ort_session = ort.InferenceSession(
                os.path.join(model_dir, 'model.onnx'), 
                providers=['CPUExecutionProvider']
            )
        else:
            self.model = BiGRUIntentModel(vocab_size=self.tokenizer.vocab_size)
            self.model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
            self.model.eval()

    def predict(self, text: str):
        input_ids = self.tokenizer.encode(text)
        
        start_time = time.perf_counter()
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: [input_ids]}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logits = torch.tensor(ort_outs[0])
        else:
            with torch.no_grad():
                logits = self.model(torch.tensor([input_ids], dtype=torch.long))
                
        latency = (time.perf_counter() - start_time) * 1000 # convert to ms
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        intent = REV_INTENT_MAP[pred_idx.item()]
        return {
            "intent": intent,
            "confidence": round(confidence.item(), 4),
            "latency_ms": round(latency, 2)
        }
if __name__ == '__main__':
    print("Initializing Inferencers...")
    inferencer_onnx = IntentInferencer(use_onnx=True)
    inferencer_pt = IntentInferencer(use_onnx=False)
    test_texts = [
        "bhai location bhej de",
        "traffic ke karan 10 minute der hogi",
        "order damage hai cancel karo",
        "customer phone ni utha rha"
    ]
    print("\n--- Testing ONNX CPU Runtime ---")
    for t in test_texts:
        res = inferencer_onnx.predict(t)
        print(f"[{t}] -> {res}")  
    print("\n--- Testing PyTorch Runtime ---")
    for t in test_texts:
        res = inferencer_pt.predict(t)
        print(f"[{t}] -> {res}")
