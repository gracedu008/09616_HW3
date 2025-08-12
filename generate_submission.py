import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from torch import nn

# Path to your best checkpoint from ESM2-medium model
CHECKPOINT_PATH = "lightning_logs/version_4/checkpoints/ESM2-medium-epoch=04-val_f1=0.9954.ckpt"  # Update this to your actual checkpoint path

# Dataset class (simplified version that focuses only on test data)
class ProteinDataset:
    def __init__(self, csv_file, label_encoder=None):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return sequence and sequence_name for test data
        sequence = self.data.iloc[idx]['sequence']
        sequence_name = self.data.iloc[idx]['sequence_name']
        return sequence, sequence_name

# Custom collate function
def collate_test_sequences(batch):
    # Separate sequences and sequence_names
    sequences = [item[0] for item in batch]
    sequence_names = [item[1] for item in batch]
    return sequences, sequence_names

# ESM2 Model class
class ESM2Classifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5, freeze_encoder=True, model_name="facebook/esm2_t12_35M_UR50D"):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, sequences):
        # Tokenize sequences
        encoded_input = self.tokenizer(
            sequences, 
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        )
        
        # Move encoded inputs to device
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        # Get sequence lengths for proper mean pooling
        seq_lengths = attention_mask.sum(dim=1).unsqueeze(1)
        
        # Get embeddings
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        sequence_output = encoder_output.last_hidden_state
        pooled_output = torch.sum(sequence_output * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        return logits
        
    def predict_step(self, batch, batch_idx):
        sequences, _ = batch
        logits = self(sequences)
        preds = torch.argmax(logits, dim=1)
        return preds

def main():
    # First, load a sample of the training data to get the label encoder
    train_data = pd.read_csv('train_data.csv')
    
    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['family_id'])
    print(f"Found {len(label_encoder.classes_)} unique protein families")
    
    # Get the list of classes for verification
    class_names = label_encoder.classes_
    print("First 5 classes:", class_names[:5])
    
    # Load the test data
    test_dataset = ProteinDataset('test_data.csv')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16,  # Larger batch size for faster inference
        shuffle=False,
        num_workers=4,
        collate_fn=collate_test_sequences
    )
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Available files in current directory:", os.listdir())
        return
    
    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    model = ESM2Classifier.load_from_checkpoint(
        CHECKPOINT_PATH,
        num_classes=len(label_encoder.classes_)
    )
    
    # Set model to evaluation mode
    model.eval()
    print("Model loaded successfully and set to evaluation mode")
    
    # Generate predictions
    print("Generating predictions...")
    sequence_names = []
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(test_loader)}")
                
            sequences, batch_sequence_names = batch
            logits = model(sequences)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            sequence_names.extend(batch_sequence_names)
    
    # Convert numeric predictions back to original labels
    predicted_classes = label_encoder.inverse_transform(predictions)
    
    # Print prediction distribution
    unique_predictions, counts = np.unique(predicted_classes, return_counts=True)
    print("\nPrediction distribution:")
    for pred_class, count in zip(unique_predictions, counts):
        print(f"{pred_class}: {count} samples")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sequence_name': sequence_names,
        'family_id': predicted_classes
    })
    
    # Check prediction diversity
    num_unique_predictions = len(submission_df['family_id'].unique())
    print(f"\nNumber of unique predicted classes: {num_unique_predictions}")
    
    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"\nSubmission file created with {len(predictions)} predictions.")
    
    # Show a sample of the predictions
    print("\nSample of predictions:")
    print(submission_df.head(10))

if __name__ == "__main__":
    import numpy as np
    main()