import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class WaveNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout_rate=0.3):
        super(WaveNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Added batch normalization
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # Embedding layer
        embeddings = self.embedding(x)
        embeddings = torch.mean(embeddings, dim=1)
        
        # Apply batch normalization
        embeddings = self.batch_norm(embeddings)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Calculate global semantics vector
        global_semantics = torch.sqrt(torch.sum(embeddings ** 2, dim=1)).unsqueeze(1)
        
        # Calculate phase vector with numerical stability
        eps = 1e-8  # Small constant for numerical stability
        phase_vector = torch.atan2(
            torch.sqrt(torch.clamp(1 - (embeddings / (global_semantics + eps)) ** 2, min=0)),
            embeddings / (global_semantics + eps)
        )

        # Create complex vector representation
        complex_vector = global_semantics * torch.exp(1j * phase_vector)

        # Wave interference with dropout
        complex_vector1 = self.dropout(self.linear1(complex_vector.real)) + 1j * self.dropout(self.linear1(complex_vector.imag))
        complex_vector2 = self.dropout(self.linear2(complex_vector.real)) + 1j * self.dropout(self.linear2(complex_vector.imag))
        interference_result = complex_vector1 + complex_vector2

        # Wave modulation
        modulation_result = complex_vector1 * complex_vector2

        # Combine results
        combined_result = interference_result + modulation_result

        # Output layer
        output = self.output_layer(combined_result.real)
        return output

# Training setup
vocab_size = 10000
embedding_dim = 768
output_dim = 4

from datasets import load_dataset
from transformers import AutoTokenizer
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Load dataset and tokenizer
dataset = load_dataset('ag_news')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create data loaders
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch])
    }

batch_size = 64
train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Training loop setup
learning_rate = 2e-3  # Slightly higher initial learning rate for OneCycleLR
num_epochs = 15  # Increased epochs
max_grad_norm = 1.0  # For gradient clipping

model = WaveNetwork(len(tokenizer), embedding_dim, output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Added weight decay

# Calculate total steps for OneCycleLR
total_steps = len(train_loader) * num_epochs

# Learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.3,  # Warm-up for 30% of training
    anneal_strategy='cos'
)

# Training loop with validation
best_accuracy = 0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        predictions = model(input_ids)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).float().mean()
        
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = epoch_acc / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(dim=1) == labels).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    val_loss = val_loss / len(test_loader)
    val_acc = val_acc / len(test_loader)
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_wave_model.pt')
    
    print(f'Epoch {epoch+1}:')
    print(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    print(f'Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
    print('-' * 50)

# Load best model for inference
model.load_state_dict(torch.load('best_wave_model.pt'))

# Inference function
def predict(model, sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    with torch.no_grad():
        predictions = model(input_ids)
    return predictions.argmax(dim=1).item()