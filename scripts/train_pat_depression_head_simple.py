#!/usr/bin/env python3
"""
Simplified PAT Depression Training Script
Directly processes NHANES data without complex extraction.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepressionHead(nn.Module):
    """Binary classification head for depression detection."""
    
    def __init__(self, input_dim: int = 96):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


def process_activity_simple(actigraphy_df: pd.DataFrame, seqn: int) -> np.ndarray:
    """Extract 7 days of activity in a simple way."""
    # Filter to subject
    subj = actigraphy_df[actigraphy_df['SEQN'] == seqn]
    
    if len(subj) < 10080:  # 7 days * 1440 minutes
        return None
        
    # Get activity counts (PAXMTSM is the main activity metric)
    activity = subj['PAXMTSM'].values[:10080]
    
    # Reshape to (7, 1440)
    activity_7d = activity.reshape(7, 1440)
    
    # Simple normalization
    activity_7d = np.clip(activity_7d, 0, 1000) / 1000.0
    
    return activity_7d.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhanes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('model_weights/pat/heads'))
    parser.add_argument('--max-subjects', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PAT encoder
    logger.info("Loading PAT model...")
    pat_model = PATModel(model_size="small")
    weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")
    if not pat_model.load_pretrained_weights(weights_path):
        logger.error("Failed to load PAT weights")
        return
        
    # Load data
    logger.info("Loading NHANES data...")
    actigraphy = pd.read_sas(args.nhanes_dir / "PAXMIN_H.xpt")
    depression = pd.read_sas(args.nhanes_dir / "DPQ_H.xpt")
    
    # Calculate PHQ-9 totals
    phq_cols = [f'DPQ0{i}0' for i in range(1, 10)]
    depression['PHQ9_TOTAL'] = depression[phq_cols].sum(axis=1)
    
    # Find subjects with both data
    common_seqns = set(actigraphy['SEQN'].unique()) & set(depression['SEQN'].unique())
    common_seqns = list(common_seqns)[:args.max_subjects]
    
    logger.info(f"Processing {len(common_seqns)} subjects...")
    
    # Extract features
    embeddings = []
    labels = []
    
    for i, seqn in enumerate(common_seqns):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(common_seqns)}")
            
        # Get activity
        activity = process_activity_simple(actigraphy, seqn)
        if activity is None:
            continue
            
        # Create PAT sequence
        from datetime import datetime, timedelta
        pat_seq = PATSequence(
            end_date=datetime.now().date(),
            activity_values=activity.flatten(),  # Flatten to (10080,)
            missing_days=[],
            data_quality_score=1.0
        )
        
        # Extract embeddings
        emb = pat_model.extract_features(pat_seq)
        if emb is None:
            continue
            
        # Get label
        phq9 = depression[depression['SEQN'] == seqn]['PHQ9_TOTAL'].iloc[0]
        if np.isnan(phq9):
            continue
            
        embeddings.append(emb)
        labels.append(1 if phq9 >= 10 else 0)
        
    logger.info(f"Extracted {len(embeddings)} samples")
    logger.info(f"Depression rate: {sum(labels)/len(labels):.2%}")
    
    # Convert to tensors
    X = torch.FloatTensor(np.array(embeddings))
    y = torch.FloatTensor(labels).unsqueeze(1)
    
    # Split data
    n_train = int(0.8 * len(X))
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    model = DepressionHead(input_dim=96)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Train
    logger.info("Training depression head...")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
                
        val_auc = roc_auc_score(val_true, val_preds)
        logger.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")
        
    # Save model
    output_path = args.output_dir / "pat_depression_head.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")
    

if __name__ == "__main__":
    main()