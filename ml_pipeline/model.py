import torch
import torch.nn as nn
class BiGRUIntentModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 5):
        super(BiGRUIntentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, input_ids: torch.Tensor):
        embedded = self.embedding(input_ids)
        outputs, _ = self.gru(embedded)
        pooled, _ = torch.max(outputs, dim=1)
        logits = self.fc(pooled)   
        return logits
