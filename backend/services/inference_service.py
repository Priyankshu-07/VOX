import os
import time
import torch
import torch.nn.functional as F
import onnxruntime as ort
import json

from .slot_extractor import SlotExtractor
from .response_generator import ResponseGenerator
from core.logger import get_logger
from core.exceptions import InferenceException

logger = get_logger("inference_service")

# Map needs to align with model training
REV_INTENT_MAP = {
    0: "get_address",
    1: "report_delay",
    2: "order_issue",
    3: "customer_unavailable",
    4: "navigation_help"
}

class InferenceEngine:
    def __init__(self):
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model'))
        self.vocab_path = os.path.join(self.model_dir, 'vocab.json')
        self.onnx_path = os.path.join(self.model_dir, 'model.onnx')
        
        self.slot_extractor = SlotExtractor()
        self.response_generator = ResponseGenerator()
        
        self.word2idx = {}
        self.max_seq_length = 20
        self.ort_session = None
        
        self.load_artifacts()

    def load_artifacts(self):
        try:
            logger.info(f"Loading vocabulary from {self.vocab_path}")
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.word2idx = data["word2idx"]
                self.max_seq_length = data.get("max_seq_length", 20)
                
            logger.info(f"Loading ONNX model from {self.onnx_path}")
            self.ort_session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
            logger.info("ML Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            raise InferenceException(f"Failed to load ML artifacts: {e}")

    def tokenize(self, text: str):
        words = text.lower().split()
        seq = [self.word2idx.get(w, self.word2idx.get("<UNK>", 1)) for w in words]
        if len(seq) < self.max_seq_length:
            seq += [self.word2idx.get("<PAD>", 0)] * (self.max_seq_length - len(seq))
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def predict(self, text: str):
        start_time = time.perf_counter()
        
        # 1. Tokenize
        input_ids = self.tokenize(text)
        
        # 2. ONNX Inference
        try:
            ort_inputs = {self.ort_session.get_inputs()[0].name: [input_ids]}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logits = torch.tensor(ort_outs[0])
        except Exception as e:
            raise InferenceException(f"ONNX Execution failed: {e}")
            
        # 3. Post-process
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        intent = REV_INTENT_MAP[pred_idx.item()]
        
        # 4. Slot Extraction
        slots = self.slot_extractor.extract(text)
        
        # 5. Generate Response
        response = self.response_generator.generate(intent, slots)
        
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Predicted intent '{intent}' in {latency:.2f}ms")
        
        return {
            "intent": intent,
            "confidence": float(confidence.item()),
            "slots": slots,
            "response": response
        }

# Singleton instance for the FastAPI application
inference_engine = InferenceEngine()
