from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchmetrics.classification import Accuracy, F1Score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

class ProteinDataset(Dataset):
    def __init__(self, csv_file, label_encoder=None, is_test=False):
        """
        Dataset class for protein sequences
        
        Args:
            csv_file: Path to CSV file with protein data
            label_encoder: LabelEncoder for family_id labels
            is_test: Whether this is test data (without labels)
        """
        self.data = pd.read_csv(csv_file)
        self.is_test = is_test
        
        # For sample_submission.csv format, we only have sequence_name and no family_id
        if 'family_id' not in self.data.columns and is_test:
            self.data['family_id'] = 'unknown'  # Placeholder for test data
        
        # Initialize label encoder if not provided
        if label_encoder is None and not is_test:
            self.label_encoder = LabelEncoder()
            self.data['encoded_label'] = self.label_encoder.fit_transform(self.data['family_id'])
            self.num_classes = len(self.label_encoder.classes_)
            print(f"Found {self.num_classes} unique protein families")
        elif not is_test:
            self.label_encoder = label_encoder
            self.data['encoded_label'] = self.label_encoder.transform(self.data['family_id'])
            self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the protein sequence
        sequence = self.data.iloc[idx]['sequence']
        
        # For test data, return only the sequence and sequence_name for later identification
        if self.is_test:
            sequence_name = self.data.iloc[idx]['sequence_name']
            return sequence, sequence_name
        else:
            # For training/validation data, return sequence and label
            label = self.data.iloc[idx]['encoded_label']
            return sequence, torch.tensor(label, dtype=torch.long)

class ProteinBERTClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5, freeze_bert=True):
        """
        Protein classification model using ProtBERT
        
        Args:
            num_classes: Number of protein family classes
            learning_rate: Learning rate for optimization
            freeze_bert: Whether to freeze the BERT encoder parameters
        """
        super().__init__()
        
        # Save hyperparameters for easy model loading
        self.save_hyperparameters()
        
        # Load pre-trained ProtBERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        # Use safe loading for ProtBERT model
        try:
            # First try with default loading
            self.bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        except Exception as e:
            print(f"Error loading model with default method: {e}")
            print("Trying alternative loading method...")
            # Try with alternative loading method
            from transformers import AutoModel
            self.bert_model = AutoModel.from_pretrained("Rostlab/prot_bert", _fast_init=False)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("BERT layers frozen for initial training")
        
        # Classification head
        bert_hidden_size = self.bert_model.config.hidden_size  # 1024 for ProtBERT
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes)
        )
        
        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
    
    def forward(self, sequences):
        # Tokenize sequences
        encoded_input = self.tokenizer(
            sequences, 
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        
        # Move encoded inputs to the same device as model
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        # Get sequence lengths for proper mean pooling
        seq_lengths = attention_mask.sum(dim=1).unsqueeze(1)
        
        # Get BERT embeddings
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use mean pooling over sequence length
        sequence_output = bert_output.last_hidden_state
        pooled_output = torch.sum(sequence_output * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        
        # Get logits through classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        train_acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        val_acc = self.val_acc(preds, labels)
        val_f1 = self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            sequences = batch[0]  # Handle validation data format
        else:
            sequences = batch  # Handle test data format
            
        logits = self(sequences)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return preds, probs
    
    def configure_optimizers(self):
        # Use different learning rates for BERT and classification head
        bert_params = list(self.bert_model.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': self.hparams.learning_rate / 10},
            {'params': classifier_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

def train_protein_classifier():
    # Load and prepare datasets
    train_dataset = ProteinDataset('train_data.csv')
    val_dataset = ProteinDataset('val_data.csv', train_dataset.label_encoder)
    test_dataset = ProteinDataset('test_data.csv', train_dataset.label_encoder, is_test=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_sequences  # Custom collate function to handle variable length sequences
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_sequences
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_test_sequences  # Different collate for test data
    )
    
    # Create model
    model = ProteinBERTClassifier(
        num_classes=train_dataset.num_classes,
        learning_rate=2e-5,  # Lower learning rate for fine-tuning
        freeze_bert=True  # Initially freeze BERT layers
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        filename='protein-bert-{epoch:02d}-{val_f1:.4f}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=3,
        mode='max'
    )
    
    # Initialize trainer for initial training with frozen BERT
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Initial training with frozen BERT
    print("Phase 1: Training with frozen BERT layers...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint from Phase 1
    best_model_path = checkpoint_callback.best_model_path
    best_f1 = checkpoint_callback.best_model_score.item()
    print(f"Phase 1 complete. Best validation F1: {best_f1:.4f}")
    
    # Phase 2: Fine-tune the entire model including BERT layers
    print("Phase 2: Fine-tuning entire model...")
    
    # Load best model from Phase 1 and unfreeze BERT
    model = ProteinBERTClassifier.load_from_checkpoint(best_model_path)
    
    # Unfreeze BERT layers
    for param in model.bert_model.parameters():
        param.requires_grad = True
    
    # Lower learning rate for fine-tuning
    model.hparams.learning_rate = 5e-6
    
    # New callbacks for Phase 2
    checkpoint_callback_phase2 = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        filename='protein-bert-finetuned-{epoch:02d}-{val_f1:.4f}'
    )
    
    early_stop_callback_phase2 = EarlyStopping(
        monitor='val_f1',
        patience=5,
        mode='max'
    )
    
    # Initialize trainer for fine-tuning
    trainer_phase2 = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback_phase2, early_stop_callback_phase2],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Fine-tune the model
    trainer_phase2.fit(model, train_loader, val_loader)
    
    # Load best fine-tuned model
    best_model_path_phase2 = checkpoint_callback_phase2.best_model_path
    best_f1_phase2 = checkpoint_callback_phase2.best_model_score.item()
    print(f"Phase 2 complete. Best validation F1: {best_f1_phase2:.4f}")
    
    # Use the best model overall
    if best_f1_phase2 > best_f1:
        print(f"Using fine-tuned model with F1: {best_f1_phase2:.4f}")
        best_model = ProteinBERTClassifier.load_from_checkpoint(best_model_path_phase2)
    else:
        print(f"Using phase 1 model with F1: {best_f1:.4f}")
        best_model = ProteinBERTClassifier.load_from_checkpoint(best_model_path)
    
    # Make predictions on test data
    print("Generating predictions on test data...")
    sequence_names = []
    predictions = []
    
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sequences, batch_sequence_names = batch
            logits = best_model(sequences)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            sequence_names.extend(batch_sequence_names)
    
    # Convert numeric predictions back to original labels
    predicted_classes = train_dataset.label_encoder.inverse_transform(predictions)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sequence_name': sequence_names,
        'family_id': predicted_classes
    })
    
    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"Predictions complete. Submission file created with {len(predictions)} predictions.")
    return best_model, train_dataset.label_encoder

# Custom collate functions
def collate_sequences(batch):
    # Separate sequences and labels
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return sequences, labels

def collate_test_sequences(batch):
    # Separate sequences and sequence_names
    sequences = [item[0] for item in batch]
    sequence_names = [item[1] for item in batch]
    return sequences, sequence_names

# Function for Bonus - Visualizing protein embeddings
def visualize_protein_embeddings(model, data_loader, label_encoder):
    """
    Extract protein embeddings from the model and visualize them using UMAP and t-SNE
    
    Args:
        model: Trained ProteinBERTClassifier model
        data_loader: DataLoader containing protein sequences and labels
        label_encoder: LabelEncoder used to convert between numeric and string labels
    """
    print("Extracting protein embeddings for visualization...")
    
    # Extract embeddings
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            sequences, batch_labels = batch
            
            # Tokenize sequences
            encoded_input = model.tokenizer(
                sequences, 
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            # Move encoded inputs to the same device as model
            input_ids = encoded_input['input_ids'].to(model.device)
            attention_mask = encoded_input['attention_mask'].to(model.device)
            
            # Get sequence lengths for proper mean pooling
            seq_lengths = attention_mask.sum(dim=1).unsqueeze(1)
            
            # Get BERT embeddings (only extract last hidden state, not using the classifier)
            bert_output = model.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use mean pooling over sequence length to get fixed-size representations
            sequence_output = bert_output.last_hidden_state
            pooled_output = torch.sum(sequence_output * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
            
            embeddings.append(pooled_output.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    # Combine all batches
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    print(f"Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    # Convert numeric labels to original class names
    label_names = label_encoder.inverse_transform(labels)
    
    # Reduce dimensionality with UMAP
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Reduce dimensionality with t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Create color palette with enough distinct colors
    from matplotlib import cm
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = cm.rainbow(np.linspace(0, 1, min(num_classes, 20)))
    
    # Plot UMAP visualization
    print("Generating UMAP visualization...")
    plt.figure(figsize=(14, 12))
    
    # Get top 20 most frequent classes for better visualization
    class_counts = pd.Series(labels).value_counts().index[:20]
    
    # Plot top 20 most frequent classes for clarity
    for i, cls in enumerate(class_counts):
        mask = labels == cls
        class_name = label_encoder.inverse_transform([cls])[0]
        plt.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            label=class_name,
            color=colors[i],
            alpha=0.7,
            s=50
        )
    
    plt.title('UMAP Visualization of Protein Family Embeddings', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig('protein_umap_visualization.png', dpi=300, bbox_inches='tight')
    
    # Plot t-SNE visualization
    print("Generating t-SNE visualization...")
    plt.figure(figsize=(14, 12))
    
    for i, cls in enumerate(class_counts):
        mask = labels == cls
        class_name = label_encoder.inverse_transform([cls])[0]
        plt.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            label=class_name,
            color=colors[i],
            alpha=0.7,
            s=50
        )
    
    plt.title('t-SNE Visualization of Protein Family Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig('protein_tsne_visualization.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved as protein_umap_visualization.png and protein_tsne_visualization.png")
    
    # Analyze the clusters
    print("\nAnalyzing protein family clusters...")
    
    # Calculate distances between points in embedding space
    from scipy.spatial.distance import pdist, squareform
    
    # Calculate pairwise distances in the UMAP embedding space
    umap_distances = squareform(pdist(umap_embeddings))
    
    # For each class, calculate mean distance to other classes
    class_relatedness = {}
    
    for cls in class_counts:
        cls_mask = labels == cls
        cls_name = label_encoder.inverse_transform([cls])[0]
        
        # Points belonging to this class
        cls_points = umap_embeddings[cls_mask]
        
        # Calculate mean position (centroid) of this class
        centroid = np.mean(cls_points, axis=0)
        
        # Find distances to other class centroids
        other_centroids = {}
        for other_cls in class_counts:
            if other_cls != cls:
                other_mask = labels == other_cls
                other_name = label_encoder.inverse_transform([other_cls])[0]
                other_centroid = np.mean(umap_embeddings[other_mask], axis=0)
                
                # Distance between centroids
                distance = np.linalg.norm(centroid - other_centroid)
                other_centroids[other_name] = distance
        
        # Sort by closest classes
        sorted_distances = sorted(other_centroids.items(), key=lambda x: x[1])
        class_relatedness[cls_name] = sorted_distances[:3]  # Keep top 3 closest classes
    
    # Save cluster analysis results
    with open('protein_cluster_analysis.txt', 'w') as f:
        f.write("Protein Family Cluster Analysis\n")
        f.write("==============================\n\n")
        f.write("Top 3 closest related families for each protein family:\n\n")
        
        for cls_name, related in class_relatedness.items():
            f.write(f"{cls_name}:\n")
            for related_name, distance in related:
                f.write(f"  - {related_name} (distance: {distance:.3f})\n")
            f.write("\n")
            
    print("Cluster analysis saved to protein_cluster_analysis.txt")

# Part 2: Try other models from Hugging Face
class ESMProteinClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5, freeze_encoder=True, model_name="facebook/esm2_t6_8M_UR50D"):
        """
        Protein classification model using ESM-2 (from Facebook/Meta)
        
        Args:
            num_classes: Number of protein family classes
            learning_rate: Learning rate for optimization
            freeze_encoder: Whether to freeze the encoder parameters
            model_name: Name of the ESM model to use
        """
        super().__init__()
        
        # Save hyperparameters for easy model loading
        self.save_hyperparameters()
        
        # Import here to avoid import errors if the model is not needed
        from transformers import AutoTokenizer, AutoModel
        
        # Load pre-trained ESM model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder parameters if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"ESM encoder layers frozen for initial training")
        
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
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
    
    def forward(self, sequences):
        # Tokenize sequences
        encoded_input = self.tokenizer(
            sequences, 
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=1024,  # ESM models can handle longer sequences
            add_special_tokens=True
        )
        
        # Move encoded inputs to the same device as model
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        # Get sequence lengths for proper mean pooling
        seq_lengths = attention_mask.sum(dim=1).unsqueeze(1)
        
        # Get ESM embeddings
        with torch.set_grad_enabled(not self.hparams.freeze_encoder):
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use mean pooling over sequence length
        sequence_output = encoder_output.last_hidden_state
        pooled_output = torch.sum(sequence_output * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        
        # Get logits through classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        train_acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        val_acc = self.val_acc(preds, labels)
        val_f1 = self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use different learning rates for encoder and classification head
        encoder_params = list(self.encoder.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.hparams.learning_rate / 10},
            {'params': classifier_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }


class ProtT5Classifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5, freeze_encoder=True, model_name="Rostlab/prot_t5_xl_uniref50"):
        """
        Protein classification model using ProtT5
        
        Args:
            num_classes: Number of protein family classes
            learning_rate: Learning rate for optimization
            freeze_encoder: Whether to freeze the encoder parameters
            model_name: Name of the ProtT5 model to use
        """
        super().__init__()
        
        # Save hyperparameters for easy model loading
        self.save_hyperparameters()
        
        # Import here to avoid import errors if the model is not needed
        from transformers import T5Tokenizer, T5EncoderModel
        
        # Load pre-trained ProtT5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.encoder.config.d_model
        
        # Freeze encoder parameters if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"ProtT5 encoder layers frozen for initial training")
        
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
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
    
    def forward(self, sequences):
        # Convert protein sequences to space-separated amino acids for T5 models
        sequences = [" ".join(list(seq)) for seq in sequences]
        
        # Tokenize sequences
        encoded_input = self.tokenizer(
            sequences, 
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        )
        
        # Move encoded inputs to the same device as model
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        # Get sequence lengths for proper mean pooling
        seq_lengths = attention_mask.sum(dim=1).unsqueeze(1)
        
        # Get ProtT5 embeddings
        with torch.set_grad_enabled(not self.hparams.freeze_encoder):
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use mean pooling over sequence length
        sequence_output = encoder_output.last_hidden_state
        pooled_output = torch.sum(sequence_output * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        
        # Get logits through classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        train_acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        val_acc = self.val_acc(preds, labels)
        val_f1 = self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use different learning rates for encoder and classification head
        encoder_params = list(self.encoder.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.hparams.learning_rate / 10},
            {'params': classifier_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

# Compare model performance
def compare_models():
    """
    Train and compare different protein language models on the classification task
    """
    print("Comparing protein language models for classification task...")
    
    # Load datasets
    train_dataset = ProteinDataset('train_data.csv')
    val_dataset = ProteinDataset('val_data.csv', train_dataset.label_encoder)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_sequences
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_sequences
    )
    
    # Define models to compare
    models = [
        # Baseline model: ProtBERT
        {
            "name": "ProtBERT",
            "class": ProteinBERTClassifier,
            "params": {
                "num_classes": train_dataset.num_classes,
                "learning_rate": 2e-5,
                "freeze_bert": True
            }
        },
        # ESM2 small model
        {
            "name": "ESM2-small",
            "class": ESMProteinClassifier,
            "params": {
                "num_classes": train_dataset.num_classes,
                "learning_rate": 2e-5,
                "freeze_encoder": True,
                "model_name": "facebook/esm2_t6_8M_UR50D"
            }
        },
        # ESM2 medium model
        {
            "name": "ESM2-medium",
            "class": ESMProteinClassifier,
            "params": {
                "num_classes": train_dataset.num_classes,
                "learning_rate": 2e-5,
                "freeze_encoder": True,
                "model_name": "facebook/esm2_t12_35M_UR50D"
            }
        },
        # ProtT5 model
        {
            "name": "ProtT5",
            "class": ProtT5Classifier,
            "params": {
                "num_classes": train_dataset.num_classes,
                "learning_rate": 2e-5,
                "freeze_encoder": True,
                "model_name": "Rostlab/prot_t5_xl_uniref50"
            }
        }
    ]
    
    # For storing results
    results = {}
    
    # Train and evaluate each model
    for model_config in models:
        print(f"\n\n{'='*50}")
        print(f"Training {model_config['name']} model...")
        print(f"{'='*50}")
        
        # Create model
        model = model_config["class"](**model_config["params"])
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename=f"{model_config['name']}-{{epoch:02d}}-{{val_f1:.4f}}"
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            patience=3,
            mode='max'
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=5,  # Fewer epochs for initial comparison
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            log_every_n_steps=10
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Track results
        results[model_config["name"]] = {
            "checkpoint": checkpoint_callback.best_model_path,
            "val_f1": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0,
            "config": model_config
        }
        
        print(f"Finished training {model_config['name']}. Best F1: {results[model_config['name']]['val_f1']:.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]["val_f1"])
    best_config = results[best_model_name]["config"]
    best_checkpoint = results[best_model_name]["checkpoint"]
    best_f1 = results[best_model_name]["val_f1"]
    
    print("\n\nModel Comparison Results:")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name}: F1 = {result['val_f1']:.4f}")
    print("-" * 50)
    print(f"Best model: {best_model_name} with F1 = {best_f1:.4f}")
    
    # Fine-tune the best model with unfrozen encoder
    print(f"\n\nFine-tuning the best model ({best_model_name})...")
    
    # Load the best model from checkpoint
    best_model_class = best_config["class"]
    best_model = best_model_class.load_from_checkpoint(best_checkpoint)
    
    # Unfreeze encoder layers
    if hasattr(best_model, 'bert_model'):
        for param in best_model.bert_model.parameters():
            param.requires_grad = True
    elif hasattr(best_model, 'encoder'):
        for param in best_model.encoder.parameters():
            param.requires_grad = True
    
    # Lower learning rate for fine-tuning
    best_model.hparams.learning_rate = 5e-6
    
    # Setup callbacks for fine-tuning
    checkpoint_callback_ft = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        filename=f"{best_model_name}-finetuned-{{epoch:02d}}-{{val_f1:.4f}}"
    )
    
    early_stop_callback_ft = EarlyStopping(
        monitor='val_f1',
        patience=5,
        mode='max'
    )
    
    # Initialize trainer for fine-tuning
    trainer_ft = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback_ft, early_stop_callback_ft],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Fine-tune the model
    trainer_ft.fit(best_model, train_loader, val_loader)
    
    # Final best model
    final_best_checkpoint = checkpoint_callback_ft.best_model_path
    final_best_f1 = checkpoint_callback_ft.best_model_score.item() if checkpoint_callback_ft.best_model_score else best_f1
    
    print(f"Fine-tuning complete. Final best F1: {final_best_f1:.4f}")
    
    # Return the best model configuration and checkpoint
    return best_model_name, final_best_checkpoint


if __name__ == "__main__":
    # Phase 1: Train baseline model
    print("Phase 1: Training baseline ProtBERT model...")
    baseline_model, label_encoder = train_protein_classifier()
    
    # Phase 2: Compare different models
    print("\nPhase 2: Comparing different protein language models...")
    best_model_name, best_checkpoint = compare_models()
    
    # Load the best model
    if "ESM" in best_model_name:
        model_class = ESMProteinClassifier
    elif "ProtT5" in best_model_name:
        model_class = ProtT5Classifier
    else:
        model_class = ProteinBERTClassifier
    
    best_model = model_class.load_from_checkpoint(best_checkpoint)
    
    # Generate test predictions from best model
    test_dataset = ProteinDataset('test_data.csv', label_encoder, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_test_sequences
    )
    
    print(f"\nGenerating test predictions using best model: {best_model_name}...")
    
    # Make predictions
    sequence_names = []
    predictions = []
    
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sequences, batch_sequence_names = batch
            logits = best_model(sequences)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            sequence_names.extend(batch_sequence_names)
    
    # Convert numeric predictions back to original labels
    predicted_classes = label_encoder.inverse_transform(predictions)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sequence_name': sequence_names,
        'family_id': predicted_classes
    })
    
    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"Predictions complete. Submission file created with {len(predictions)} predictions.")
    
    # Bonus: Create protein embeddings visualization
    print("\nBonus: Creating protein embedding visualizations...")
    
    # Create validation dataset and loader for visualization
    val_dataset = ProteinDataset('val_data.csv', label_encoder)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False,
        collate_fn=collate_sequences
    )
    
    # Visualize embeddings
    visualize_protein_embeddings(best_model, val_loader, label_encoder)