"""
Advanced Training Script for 5G Anomaly Detection with Epochs
==============================================================

Features:
- Loads data from CSV file
- Train/validation/test split
- Multiple epochs with early stopping
- Displays accuracy per epoch
- Saves best model
- Shows confusion matrix and classification report
- Visualizes training progress

Author: Industrial IoT Monitoring System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "data/5g_training_data_labeled.csv"
MODEL_OUTPUT_PATH = "models/5g_supervised_model.pkl"
EPOCHS = 10
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

class EpochBasedTrainer:
    """Train Random Forest with epoch-based approach"""
    
    def __init__(self, n_estimators_per_epoch=50, max_epochs=10):
        self.n_estimators_per_epoch = n_estimators_per_epoch
        self.max_epochs = max_epochs
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        self.best_model = None
        self.best_val_acc = 0
        self.best_epoch = 0
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train model with multiple epochs"""
        
        print("\n" + "=" * 70)
        print("EPOCH-BASED TRAINING")
        print("=" * 70)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"\n🔧 Training Configuration:")
        print(f"   Estimators per epoch: {self.n_estimators_per_epoch}")
        print(f"   Maximum epochs: {self.max_epochs}")
        print(f"   Total trees (max): {self.n_estimators_per_epoch * self.max_epochs}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        
        print(f"\n📊 Starting Training...")
        print("-" * 70)
        
        for epoch in range(1, self.max_epochs + 1):
            # Calculate total estimators up to this epoch
            n_estimators = self.n_estimators_per_epoch * epoch
            
            # Train Random Forest with increasing trees
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                warm_start=False  # Train fresh each time
            )
            
            self.model.fit(X_train_scaled, y_train_encoded)
            
            # Predictions
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train_encoded, train_pred)
            val_acc = accuracy_score(y_val_encoded, val_pred)
            
            # Calculate "loss" as 1 - accuracy (for visualization)
            train_loss = 1 - train_acc
            val_loss = 1 - val_acc
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model = self.model
                best_marker = " 🌟 NEW BEST!"
            else:
                best_marker = ""
            
            # Print epoch results
            print(f"Epoch {epoch:2d}/{self.max_epochs} | "
                  f"Trees: {n_estimators:3d} | "
                  f"Train Acc: {train_acc*100:6.2f}% | "
                  f"Val Acc: {val_acc*100:6.2f}% | "
                  f"Val Loss: {val_loss:.4f}{best_marker}")
        
        print("-" * 70)
        print(f"\n✅ Training Complete!")
        print(f"   Best Epoch: {self.best_epoch}")
        print(f"   Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"   Best Model Trees: {self.n_estimators_per_epoch * self.best_epoch}")
        
        # Use best model
        self.model = self.best_model
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        
        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled)
        
        test_acc = accuracy_score(y_test_encoded, predictions)
        test_precision = precision_score(y_test_encoded, predictions, average='weighted')
        test_recall = recall_score(y_test_encoded, predictions, average='weighted')
        test_f1 = f1_score(y_test_encoded, predictions, average='weighted')
        
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:  {test_acc*100:.2f}%")
        print(f"   Precision: {test_precision*100:.2f}%")
        print(f"   Recall:    {test_recall*100:.2f}%")
        print(f"   F1-Score:  {test_f1*100:.2f}%")
        
        # Classification Report
        print(f"\n📋 Detailed Classification Report:")
        print("-" * 70)
        report = classification_report(
            y_test_encoded, 
            predictions,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_encoded, predictions)
        
        return {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': y_test_encoded
        }
    
    def plot_training_history(self):
        """Visualize training progress"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Accuracy
        axes[0].plot(self.history['epoch'], 
                     [acc * 100 for acc in self.history['train_acc']], 
                     'b-o', label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history['epoch'], 
                     [acc * 100 for acc in self.history['val_acc']], 
                     'r-o', label='Validation Accuracy', linewidth=2)
        axes[0].axvline(x=self.best_epoch, color='green', linestyle='--', 
                       label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        
        # Plot Loss
        axes[1].plot(self.history['epoch'], self.history['train_loss'], 
                     'b-o', label='Training Loss', linewidth=2)
        axes[1].plot(self.history['epoch'], self.history['val_loss'], 
                     'r-o', label='Validation Loss', linewidth=2)
        axes[1].axvline(x=self.best_epoch, color='green', linestyle='--', 
                       label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Training history plot saved to: models/training_history.png")
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap"""
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix plot saved to: models/confusion_matrix.png")
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model and preprocessing objects"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'n_estimators': self.model.n_estimators,
            'anomaly_classes': list(self.label_encoder.classes_),
            'training_history': self.history,
            'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved successfully to: {filepath}")
        print(f"   Model Type: Random Forest Classifier")
        print(f"   Total Trees: {self.model.n_estimators}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        print(f"   Best Validation Accuracy: {self.best_val_acc*100:.2f}%")

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("5G BASE STATION ANOMALY DETECTION - ADVANCED TRAINING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print(f"\n📁 Loading data from: {DATA_FILE}")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file not found!")
        print(f"   Please run 'python scripts/generate_training_data.py' first")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Show class distribution
    print(f"\n📊 Class Distribution:")
    class_counts = df['anomaly_type'].value_counts()
    for anomaly_type, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {anomaly_type:20s}: {count:5,} ({percentage:5.2f}%)")
    
    # Prepare features and labels
    feature_cols = ['temperature_C', 'power_consumption_W', 
                   'signal_strength_dBm', 'network_load_percent']
    
    X = df[feature_cols].values
    y = df['anomaly_type'].values
    
    print(f"\n🔧 Data Preparation:")
    print(f"   Features: {', '.join(feature_cols)}")
    print(f"   Total samples: {len(X):,}")
    print(f"   Feature shape: {X.shape}")
    
    # Split data: Train / Validation / Test
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"   Training set:   {len(X_train):5,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation set: {len(X_val):5,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test set:       {len(X_test):5,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Initialize trainer
    trainer = EpochBasedTrainer(
        n_estimators_per_epoch=50,
        max_epochs=EPOCHS
    )
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_results['confusion_matrix'])
    
    # Save model
    trainer.save_model(MODEL_OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results Summary:")
    print(f"   Best Epoch: {trainer.best_epoch}/{EPOCHS}")
    print(f"   Validation Accuracy: {trainer.best_val_acc*100:.2f}%")
    print(f"   Test Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"   Test Precision: {test_results['precision']*100:.2f}%")
    print(f"   Test Recall: {test_results['recall']*100:.2f}%")
    print(f"   Test F1-Score: {test_results['f1_score']*100:.2f}%")
    print(f"\n💾 Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"📈 Training plots saved to: models/")
    print(f"\n🚀 You can now run the Streamlit dashboard:")
    print(f"   python scripts/run_supervised_dashboard.py")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
