"""
⚖️ SUPERVISED vs UNSUPERVISED LEARNING COMPARISON
==================================================

This script compares both approaches side-by-side on the same test data.

Usage:
    python compare_models.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import supervised version
from realtime_5g_simulator_supervised import FiveGBaseStation, SupervisedAnomalyDetector

# Import unsupervised version (we'll recreate it here for comparison)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


class UnsupervisedDetector:
    """Simplified unsupervised detector for comparison"""
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.05,
            n_estimators=200,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = ['temperature_C', 'power_consumption_W', 
                            'signal_strength_dBm', 'network_load_percent']
    
    def train(self, training_data):
        """Train on unlabeled data"""
        df = pd.DataFrame(training_data)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
    
    def predict(self, data):
        """Predict: -1 = anomaly, 1 = normal"""
        features = np.array([[
            data['temperature_C'],
            data['power_consumption_W'],
            data['signal_strength_dBm'],
            data['network_load_percent']
        ]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        score = self.model.score_samples(features_scaled)[0]
        
        is_anomaly = (prediction == -1)
        return is_anomaly, score


def generate_test_data(n_samples=200):
    """Generate labeled test dataset"""
    print(f"📊 Generating {n_samples} test samples...")
    simulator = FiveGBaseStation("TEST-STATION")
    test_data = []
    
    for _ in range(n_samples):
        test_data.append(simulator.generate_sensor_data())
    
    df = pd.DataFrame(test_data)
    print(f"\n✅ Generated {len(test_data)} samples")
    print("\n📊 Test Data Distribution:")
    print(df['anomaly_label'].value_counts().sort_index())
    
    return test_data


def evaluate_models():
    """Compare supervised vs unsupervised learning"""
    
    print("⚖️  SUPERVISED vs UNSUPERVISED LEARNING COMPARISON")
    print("=" * 70)
    print()
    
    # Generate test data
    test_data = generate_test_data(200)
    
    # Initialize models
    print("\n🤖 Initializing models...")
    supervised_detector = SupervisedAnomalyDetector()
    unsupervised_detector = UnsupervisedDetector()
    
    # Train unsupervised model (it doesn't use labels)
    print("🔄 Training unsupervised model (Isolation Forest)...")
    unsupervised_detector.train(test_data)
    
    print("✅ Supervised model already trained")
    print("✅ Unsupervised model trained")
    
    # Evaluate both models
    print("\n🧪 Evaluating on test data...")
    
    supervised_results = []
    unsupervised_results = []
    
    for data in test_data:
        # Supervised prediction
        sup_pred, sup_conf = supervised_detector.predict(data)
        supervised_results.append({
            'actual': data['anomaly_label'],
            'predicted': sup_pred,
            'confidence': sup_conf,
            'is_correct': (sup_pred == data['anomaly_label'])
        })
        
        # Unsupervised prediction
        unsup_pred, unsup_score = unsupervised_detector.predict(data)
        actual_is_anomaly = (data['anomaly_label'] != 'normal')
        unsupervised_results.append({
            'actual': data['anomaly_label'],
            'predicted': 'anomaly' if unsup_pred else 'normal',
            'score': unsup_score,
            'is_correct': (unsup_pred == actual_is_anomaly)
        })
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    # Supervised metrics
    sup_correct = sum(1 for r in supervised_results if r['is_correct'])
    sup_accuracy = (sup_correct / len(supervised_results)) * 100
    
    print("\n🎯 SUPERVISED LEARNING (Random Forest Classifier)")
    print("-" * 70)
    print(f"   Algorithm: Random Forest with 200 trees")
    print(f"   Output: Multi-class (7 classes)")
    print(f"   Accuracy: {sup_accuracy:.2f}%")
    print(f"   Correct predictions: {sup_correct}/{len(supervised_results)}")
    
    # Per-class accuracy for supervised
    print("\n   Per-Class Performance:")
    df_sup = pd.DataFrame(supervised_results)
    for class_name in supervised_detector.anomaly_classes:
        class_data = df_sup[df_sup['actual'] == class_name]
        if len(class_data) > 0:
            class_acc = (class_data['is_correct'].sum() / len(class_data)) * 100
            print(f"     • {class_name.upper():20s}: {class_acc:5.1f}% ({class_data['is_correct'].sum}/{len(class_data)})")
    
    # Unsupervised metrics
    unsup_correct = sum(1 for r in unsupervised_results if r['is_correct'])
    unsup_accuracy = (unsup_correct / len(unsupervised_results)) * 100
    
    print("\n🔍 UNSUPERVISED LEARNING (Isolation Forest)")
    print("-" * 70)
    print(f"   Algorithm: Isolation Forest")
    print(f"   Output: Binary (normal vs anomaly)")
    print(f"   Accuracy: {unsup_accuracy:.2f}%")
    print(f"   Correct predictions: {unsup_correct}/{len(unsupervised_results)}")
    
    # Binary classification breakdown
    df_unsup = pd.DataFrame(unsupervised_results)
    
    # True positives (correctly detected anomalies)
    true_positives = sum(1 for r in unsupervised_results 
                        if r['predicted'] == 'anomaly' and r['actual'] != 'normal')
    
    # True negatives (correctly identified normal)
    true_negatives = sum(1 for r in unsupervised_results 
                        if r['predicted'] == 'normal' and r['actual'] == 'normal')
    
    # False positives (normal flagged as anomaly)
    false_positives = sum(1 for r in unsupervised_results 
                         if r['predicted'] == 'anomaly' and r['actual'] == 'normal')
    
    # False negatives (anomaly missed)
    false_negatives = sum(1 for r in unsupervised_results 
                         if r['predicted'] == 'normal' and r['actual'] != 'normal')
    
    print(f"\n   Confusion Matrix:")
    print(f"     • True Positives (anomaly detected):  {true_positives}")
    print(f"     • True Negatives (normal identified): {true_negatives}")
    print(f"     • False Positives (false alarms):     {false_positives}")
    print(f"     • False Negatives (missed anomalies): {false_negatives}")
    
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"\n     Precision: {precision*100:.2f}%")
    
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"     Recall: {recall*100:.2f}%")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("🏆 SUMMARY")
    print("=" * 70)
    
    if sup_accuracy > unsup_accuracy:
        winner = "SUPERVISED"
        diff = sup_accuracy - unsup_accuracy
    else:
        winner = "UNSUPERVISED"
        diff = unsup_accuracy - sup_accuracy
    
    print(f"\n   Winner: {winner} (+{diff:.2f}% accuracy)")
    
    print("\n   Key Differences:")
    print("   ┌─────────────────────────┬──────────────────┬────────────────────┐")
    print("   │ Feature                 │ Supervised       │ Unsupervised       │")
    print("   ├─────────────────────────┼──────────────────┼────────────────────┤")
    print(f"   │ Accuracy                │ {sup_accuracy:6.2f}%          │ {unsup_accuracy:6.2f}%            │")
    print("   │ Anomaly Type Detection  │ Yes (7 classes)  │ No (binary only)   │")
    print("   │ Confidence Scores       │ Yes (0-100%)     │ Anomaly score      │")
    print("   │ Training Data           │ Labeled          │ Unlabeled          │")
    print("   │ Unknown Patterns        │ May miss         │ Can detect         │")
    print("   └─────────────────────────┴──────────────────┴────────────────────┘")
    
    print("\n📌 Recommendations:")
    if sup_accuracy > 90:
        print("   ✅ SUPERVISED learning performs excellently!")
        print("   → Use for production if you have labeled historical data")
        print("   → Provides actionable insights (specific anomaly types)")
    
    if unsup_accuracy > 85:
        print("   ✅ UNSUPERVISED learning performs well!")
        print("   → Good for detecting unknown patterns")
        print("   → Useful as a safety net alongside supervised")
    
    print("\n💡 Best Practice:")
    print("   Run BOTH models in parallel:")
    print("   • Supervised: Identifies known anomaly types")
    print("   • Unsupervised: Catches unknown patterns")
    print("   • Hybrid approach gives best of both worlds!")
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_models()
