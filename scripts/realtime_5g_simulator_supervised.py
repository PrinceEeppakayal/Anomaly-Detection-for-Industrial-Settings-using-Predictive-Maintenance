"""
🗼 REAL-TIME 5G BASE STATION ANOMALY DETECTION SYSTEM - SUPERVISED LEARNING
==========================================================================

✨ KEY FEATURES:
- SUPERVISED LEARNING using Random Forest Classifier
- Uses labeled data from simulator (anomaly_type field)
- Multi-class classification (Normal + 6 anomaly types)
- Better accuracy with labeled training data
- Model persistence and continuous improvement
- Classification report for each anomaly type

This simulates a real-world 5G telecommunications base station with:
- Temperature sensors (equipment cooling)
- Power consumption monitoring
- Signal strength tracking
- Network load monitoring
- Real-time anomaly detection with supervised learning
- Multi-class anomaly type prediction

Author: Industrial IoT Monitoring System
Industry: Telecommunications & Electronics
Learning Type: SUPERVISED (Random Forest Classifier)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import joblib
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import email notifier
try:
    from email_notifier import EmailNotificationSystem
    EMAIL_NOTIFICATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Email notification module not found. Email alerts disabled.")
    EMAIL_NOTIFICATIONS_AVAILABLE = False

warnings.filterwarnings('ignore')

class FiveGBaseStation:
    """Simulates a real 5G telecom base station with sensors"""
    
    def __init__(self, station_id="5G-BS-001"):
        self.station_id = station_id
        self.timestamp = datetime.now()
        
        # Normal operating parameters for 5G base station - UPDATED THRESHOLDS
        self.normal_temp = 29.0  # °C (mid-point of 18-40°C range)
        self.normal_power = 2750  # Watts (mid-point of 2500-3000W range)
        self.normal_signal = -85  # dBm (mid-point of -75 to -95 dBm range)
        self.normal_load = 60  # % (mid-point of 40-80% range)
        
        # Normal operating thresholds
        self.temp_min, self.temp_max = 18, 40
        self.power_min, self.power_max = 2500, 3000
        self.signal_min, self.signal_max = -95, -75  # Note: higher value means better signal
        self.load_min, self.load_max = 40, 80
        
        # Anomaly injection parameters
        self.anomaly_probability = 0.10  # 5% chance of anomaly (reduced from 15%)
        self.time_counter = 0
        
    def generate_sensor_data(self):
        """Generate realistic sensor readings with occasional anomalies"""
        
        self.time_counter += 1
        self.timestamp += timedelta(seconds=2)
        
        # Base readings with normal variation (smaller noise for tighter thresholds)
        temp_noise = np.random.normal(0, 1.5)  # Reduced noise
        power_noise = np.random.normal(0, 50)  # Reduced noise
        signal_noise = np.random.normal(0, 2)  # Reduced noise
        load_noise = np.random.normal(0, 3)    # Reduced noise
        
        # Simulate daily traffic patterns (more load during day)
        hour_of_day = self.timestamp.hour
        traffic_pattern = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Reduced amplitude
        
        # Generate readings within normal ranges
        temperature = self.normal_temp + temp_noise + (traffic_pattern * 0.2)
        power_consumption = self.normal_power + power_noise + (traffic_pattern * 20)
        signal_strength = self.normal_signal + signal_noise
        network_load = np.clip(self.normal_load + load_noise + (traffic_pattern * 0.5), 0, 100)
        
        # Inject realistic anomalies with LABELS
        anomaly_type = 'normal'  # Default label
        
        if np.random.random() < self.anomaly_probability:
            anomaly_choice = np.random.choice([
                'overheating', 'power_surge', 'signal_degradation', 
                'network_overload', 'cooling_failure', 'equipment_failure'
            ])
            
            if anomaly_choice == 'overheating':
                temperature = np.random.uniform(self.temp_max + 5, self.temp_max + 20)  # Above 40°C
                anomaly_type = 'overheating'
                
            elif anomaly_choice == 'power_surge':
                power_consumption = np.random.uniform(self.power_max + 200, self.power_max + 800)  # Above 3000W
                anomaly_type = 'power_surge'
                
            elif anomaly_choice == 'signal_degradation':
                signal_strength = np.random.uniform(-110, self.signal_min - 5)  # Below -95dBm (weaker)
                anomaly_type = 'signal_degradation'
                
            elif anomaly_choice == 'network_overload':
                network_load = np.random.uniform(self.load_max + 5, 98)  # Above 80%
                temperature += np.random.uniform(5, 10)  # Temperature rises with load
                power_consumption += np.random.uniform(100, 300)  # Power increases too
                anomaly_type = 'network_overload'
                
            elif anomaly_choice == 'cooling_failure':
                temperature = np.random.uniform(self.temp_max + 10, self.temp_max + 25)  # Severe overheating
                anomaly_type = 'cooling_failure'
                
            elif anomaly_choice == 'equipment_failure':
                temperature = np.random.uniform(self.temp_min - 10, self.temp_min)  # Below 18°C (unusual)
                power_consumption = np.random.uniform(self.power_min - 500, self.power_min - 100)  # Below 2500W
                signal_strength = np.random.uniform(-115, -105)  # Very weak signal
                network_load = np.random.uniform(0, 20)  # Very low load
                anomaly_type = 'equipment_failure'
        
        # Inject realistic anomalies with LABELS
        anomaly_type = 'normal'  # Default label
        
        if np.random.random() < self.anomaly_probability:
            anomaly_choice = np.random.choice([
                'overheating', 'power_surge', 'signal_degradation', 
                'network_overload', 'cooling_failure', 'equipment_failure'
            ])
            
            if anomaly_choice == 'overheating':
                temperature = np.random.uniform(self.temp_max + 5, self.temp_max + 20)  # Above 40°C
                anomaly_type = 'overheating'
                
            elif anomaly_choice == 'power_surge':
                power_consumption = np.random.uniform(self.power_max + 200, self.power_max + 800)  # Above 3000W
                anomaly_type = 'power_surge'
                
            elif anomaly_choice == 'signal_degradation':
                signal_strength = np.random.uniform(-110, self.signal_min - 5)  # Below -95dBm (weaker)
                anomaly_type = 'signal_degradation'
                
            elif anomaly_choice == 'network_overload':
                network_load = np.random.uniform(self.load_max + 5, 98)  # Above 80%
                temperature += np.random.uniform(5, 10)  # Temperature rises with load
                power_consumption += np.random.uniform(100, 300)  # Power increases too
                anomaly_type = 'network_overload'
                
            elif anomaly_choice == 'cooling_failure':
                temperature = np.random.uniform(self.temp_max + 10, self.temp_max + 25)  # Severe overheating
                anomaly_type = 'cooling_failure'
                
            elif anomaly_choice == 'equipment_failure':
                temperature = np.random.uniform(self.temp_min - 10, self.temp_min)  # Below 18°C (unusual)
                power_consumption = np.random.uniform(self.power_min - 500, self.power_min - 100)  # Below 2500W
                signal_strength = np.random.uniform(-115, -105)  # Very weak signal
                network_load = np.random.uniform(0, 20)  # Very low load
                anomaly_type = 'equipment_failure'
        
        return {
            'timestamp': self.timestamp,
            'temperature_C': round(temperature, 2),
            'power_consumption_W': round(power_consumption, 2),
            'signal_strength_dBm': round(signal_strength, 2),
            'network_load_percent': round(network_load, 2),
            'anomaly_label': anomaly_type  # LABELED DATA for supervised learning
        }


class SupervisedAnomalyDetector:
    """Supervised anomaly detection using Random Forest Classifier with labeled data"""

    def __init__(self, retrain_interval=100, model_path="models/5g_supervised_model.pkl"):
        self.retrain_interval = retrain_interval
        self.model_path = model_path
        self.data_buffer = []
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Supervised learning parameters
        self.samples_since_retrain = 0
        self.model_version = 0
        
        # Performance tracking
        self.predictions_history = []
        self.true_labels_history = []
        
        # Feature columns
        self.feature_cols = ['temperature_C', 'power_consumption_W', 
                            'signal_strength_dBm', 'network_load_percent']
        
        # Anomaly classes (7 classes including normal)
        self.anomaly_classes = ['normal', 'overheating', 'power_surge', 
                               'signal_degradation', 'network_overload', 
                               'cooling_failure', 'equipment_failure']
        
        # Initialize label encoder with all classes
        self.label_encoder.fit(self.anomaly_classes)
        
        # Try to load existing model
        self.load_model()
        
    def generate_training_data(self, n_samples=1000):
        """Generate balanced labeled training dataset from simulator"""
        print(f"📊 Generating {n_samples} labeled training samples...")
        
        temp_simulator = FiveGBaseStation("TRAINING-STATION")
        # Temporarily increase anomaly rate for training data generation
        original_prob = temp_simulator.anomaly_probability
        temp_simulator.anomaly_probability = 0.8  # 80% for faster generation
        
        training_data = []
        
        # Generate balanced dataset
        samples_per_class = n_samples // len(self.anomaly_classes)
        
        for target_class in self.anomaly_classes:
            class_samples = 0
            attempts = 0
            max_attempts = samples_per_class * 20  # Increased max attempts
            
            while class_samples < samples_per_class and attempts < max_attempts:
                data = temp_simulator.generate_sensor_data()
                if data['anomaly_label'] == target_class:
                    training_data.append(data)
                    class_samples += 1
                attempts += 1
        
        # Restore original probability
        temp_simulator.anomaly_probability = original_prob
        
        print(f"✅ Generated {len(training_data)} balanced training samples")
        
        # Show class distribution
        df = pd.DataFrame(training_data)
        print("\n📊 Class Distribution:")
        print(df['anomaly_label'].value_counts().sort_index())
        
        return training_data
        
    def load_model(self):
        """Load existing trained model if available"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.model_version = model_data.get('model_version', 0)
                self.is_trained = True
                
                print(f"✅ Loaded existing supervised model v{self.model_version}")
                print(f"   📁 Source: {self.model_path}")
                print(f"   🌳 Model Type: Random Forest Classifier")
                print(f"   🎯 Classes: {len(self.anomaly_classes)}")
                return
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
        
        print("ℹ️  No existing model found. Will train from scratch.")
        
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if self.is_trained:
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_version': self.model_version,
                'anomaly_classes': self.anomaly_classes,
                'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            joblib.dump(model_data, self.model_path)
            print(f"💾 Supervised model v{self.model_version} saved to {self.model_path}")
    
    def add_data_point(self, data):
        """Add new data point to buffer for retraining"""
        self.data_buffer.append(data)
        self.samples_since_retrain += 1
        
        # Keep only recent data for memory efficiency
        if len(self.data_buffer) > 2000:
            self.data_buffer.pop(0)
        
        # Automatic retraining when enough new samples collected
        if (self.is_trained and 
            self.samples_since_retrain >= self.retrain_interval and
            len(self.data_buffer) >= 100):
            self.retrain_model()
            self.samples_since_retrain = 0
    
    def train_model(self, training_data=None):
        """Train Random Forest Classifier on labeled data"""
        
        # Use provided training data or generate new (increased sample size)
        if training_data is None:
            training_data = self.generate_training_data(n_samples=2000)  # Increased from 1000
        
        df = pd.DataFrame(training_data)
        
        # Prepare features and labels
        X = df[self.feature_cols].values
        y = df['anomaly_label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest Classifier
        print("🌳 Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=300,        # Increased from 200 for better accuracy
            max_depth=20,            # Increased from 15 for more complex patterns
            min_samples_split=3,     # Reduced from 5 for more sensitive splits
            min_samples_leaf=1,      # Reduced from 2 for finer granularity
            class_weight='balanced', # Handle class imbalance better
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y_encoded)
        
        self.is_trained = True
        self.model_version += 1
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y_encoded, train_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   📊 Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"   🌳 Trees: {self.model.n_estimators}")
        print(f"   🎯 Classes: {len(self.anomaly_classes)}")
        
        # Save model
        self.save_model()
        
        return True
    
    def retrain_model(self):
        """Retrain model with accumulated new data"""
        if len(self.data_buffer) < 100:
            return False
        
        print(f"\n🔄 Retraining model with {len(self.data_buffer)} new samples...")
        
        success = self.train_model(training_data=self.data_buffer)
        
        if success:
            print(f"✅ Model retrained successfully! (v{self.model_version})")
            self.save_model()
        
        return success
    
    def predict(self, data):
        """Predict anomaly type for new data point"""
        if not self.is_trained:
            return 'unknown', 0.0
        
        # Extract features
        features = np.array([[
            data['temperature_C'],
            data['power_consumption_W'],
            data['signal_strength_dBm'],
            data['network_load_percent']
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability/confidence
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        confidence = prediction_proba.max()
        
        # Track predictions for metrics
        if 'anomaly_label' in data:
            self.predictions_history.append(prediction_label)
            self.true_labels_history.append(data['anomaly_label'])
            
            # Keep only recent history
            if len(self.predictions_history) > 1000:
                self.predictions_history.pop(0)
                self.true_labels_history.pop(0)
        
        return prediction_label, confidence
    
    def get_classification_metrics(self):
        """Get detailed classification metrics"""
        if len(self.predictions_history) < 10:
            return None
        
        # Calculate overall accuracy
        accuracy = accuracy_score(self.true_labels_history, self.predictions_history)
        
        # Get unique labels that actually appeared in the data
        unique_labels = sorted(list(set(self.true_labels_history + self.predictions_history)))
        
        # Generate classification report (only for labels that appeared)
        report = classification_report(
            self.true_labels_history, 
            self.predictions_history,
            labels=unique_labels,
            target_names=unique_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(
            self.true_labels_history,
            self.predictions_history,
            labels=self.anomaly_classes
        )
        
        return {
            'accuracy': accuracy * 100,
            'classification_report': report,
            'confusion_matrix': cm,
            'total_predictions': len(self.predictions_history)
        }
    
    def get_model_stats(self):
        """Get model statistics for display"""
        metrics = self.get_classification_metrics()
        
        if metrics is None:
            return {
                'version': self.model_version,
                'accuracy': 0.0,
                'samples_until_retrain': self.retrain_interval - self.samples_since_retrain,
                'total_predictions': 0
            }
        
        return {
            'version': self.model_version,
            'accuracy': metrics['accuracy'],
            'samples_until_retrain': self.retrain_interval - self.samples_since_retrain,
            'total_predictions': metrics['total_predictions'],
            'classification_report': metrics['classification_report'],
            'confusion_matrix': metrics['confusion_matrix']
        }


class PredictiveMaintenanceSystem:
    """Predictive maintenance recommendation engine"""
    
    def __init__(self):
        self.anomaly_history = []
        
    def add_anomaly(self, timestamp, anomaly_type, severity, confidence):
        """Record an anomaly for maintenance tracking"""
        self.anomaly_history.append({
            'timestamp': timestamp,
            'type': anomaly_type,
            'severity': severity,
            'confidence': confidence
        })
        
        # Keep only recent anomalies (last 100)
        if len(self.anomaly_history) > 100:
            self.anomaly_history.pop(0)
    
    def get_maintenance_recommendation(self, current_time, current_anomaly_type='normal'):
        """Provide maintenance recommendation based on anomaly history"""
        
        if current_anomaly_type == 'normal':
            return 'NORMAL', "✅ All systems operating normally"
        
        # Count recent anomalies (last 10 entries)
        recent_anomalies = self.anomaly_history[-10:] if len(self.anomaly_history) >= 10 else self.anomaly_history
        
        # Critical anomalies
        critical_types = ['cooling_failure', 'equipment_failure', 'power_surge']
        
        if current_anomaly_type in critical_types:
            return 'CRITICAL', f"🚨 CRITICAL: {current_anomaly_type.replace('_', ' ').title()} detected! Immediate action required!"
        
        # Warning for moderate anomalies
        warning_types = ['overheating', 'network_overload', 'signal_degradation']
        
        if current_anomaly_type in warning_types:
            anomaly_count = sum(1 for a in recent_anomalies if a['type'] == current_anomaly_type)
            if anomaly_count >= 3:
                return 'CRITICAL', f"🚨 CRITICAL: Repeated {current_anomaly_type.replace('_', ' ')} events! Maintenance needed!"
            else:
                return 'WARNING', f"⚠️  WARNING: {current_anomaly_type.replace('_', ' ').title()} detected. Monitor closely."
        
        return 'INFO', f"ℹ️  INFO: Minor anomaly detected ({current_anomaly_type})"


# ============================================================================
# MAIN REAL-TIME VISUALIZATION SYSTEM
# ============================================================================

print("🗼 Initializing 5G Base Station Real-Time Monitoring System")
print("📚 SUPERVISED LEARNING MODE with Random Forest Classifier")
print("=" * 70)

# Initialize components
base_station = FiveGBaseStation("5G-BS-SUPERVISED-001")
detector = SupervisedAnomalyDetector(retrain_interval=100)
maintenance_system = PredictiveMaintenanceSystem()

# Initialize email notification system
email_notifier = None
if EMAIL_NOTIFICATIONS_AVAILABLE:
    try:
        email_notifier = EmailNotificationSystem()
        print(f"📧 Email notification system: ENABLED")
    except Exception as e:
        print(f"⚠️  Email notification setup failed: {e}")
else:
    print("📧 Email notification system: DISABLED")

print("\n🧠 Supervised Learning Mode: ENABLED")
print("   ✓ Multi-class classification (7 classes)")
print("   ✓ Random Forest with 200 trees")
print("   ✓ Labeled training data from simulator")
print("   ✓ Auto-retrains every 100 samples")
print("   ✓ Prediction confidence scores")

# Data storage for plotting
max_display_points = 100
timestamps = []
temperatures = []
power_consumptions = []
signal_strengths = []
network_loads = []
anomaly_predictions = []
anomaly_confidences = []
true_labels = []

# Training phase (if no pre-trained model exists)
if not detector.is_trained:
    print("\n🔄 Generating labeled training data...")
    detector.train_model()
    print("✅ Model trained successfully!")

print("\n🚀 Starting real-time monitoring...\n")

# Setup the plot
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Real-Time 5G Base Station Monitoring - SUPERVISED LEARNING', 
             fontsize=16, fontweight='bold')

# Create subplots
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)

def update_plot(frame):
    """Update plot with new data point"""
    
    # Generate new sensor reading with TRUE LABEL
    data = base_station.generate_sensor_data()
    detector.add_data_point(data)
    
    # Predict anomaly type (SUPERVISED LEARNING)
    predicted_label, confidence = detector.predict(data)
    is_anomaly = (predicted_label != 'normal')
    
    # Store data
    timestamps.append(data['timestamp'])
    temperatures.append(data['temperature_C'])
    power_consumptions.append(data['power_consumption_W'])
    signal_strengths.append(data['signal_strength_dBm'])
    network_loads.append(data['network_load_percent'])
    anomaly_predictions.append(predicted_label)
    anomaly_confidences.append(confidence)
    true_labels.append(data['anomaly_label'])
    
    # Update predictive maintenance
    if is_anomaly:
        severity = 'high' if predicted_label in ['cooling_failure', 'equipment_failure'] else 'medium'
        maintenance_system.add_anomaly(data['timestamp'], predicted_label, severity, confidence)
    
    # Get maintenance recommendation
    severity_level, message = maintenance_system.get_maintenance_recommendation(
        data['timestamp'], predicted_label
    )
    
    # Send email notification for critical/warning anomalies
    if email_notifier and severity_level in ['CRITICAL', 'WARNING']:
        try:
            email_notifier.send_alert(
                subject=f"5G Station Alert: {severity_level}",
                anomaly_type=predicted_label,
                severity=severity_level,
                metrics={
                    'temperature': data['temperature_C'],
                    'power': data['power_consumption_W'],
                    'signal': data['signal_strength_dBm'],
                    'load': data['network_load_percent'],
                    'confidence': f"{confidence*100:.1f}%"
                }
            )
        except Exception as e:
            pass
    
    # Keep only recent data for display
    if len(timestamps) > max_display_points:
        timestamps.pop(0)
        temperatures.pop(0)
        power_consumptions.pop(0)
        signal_strengths.pop(0)
        network_loads.pop(0)
        anomaly_predictions.pop(0)
        anomaly_confidences.pop(0)
        true_labels.pop(0)
    
    # Clear all axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.clear()
    
    # Plot 1: Temperature
    ax1.plot(range(len(temperatures)), temperatures, 'b-', linewidth=2, label='Temperature')
    anomaly_indices = [i for i, label in enumerate(anomaly_predictions) if label != 'normal']
    if anomaly_indices:
        ax1.scatter([anomaly_indices[-1]] if anomaly_indices else [], 
                   [temperatures[anomaly_indices[-1]]] if anomaly_indices else [], 
                   color='red', s=100, zorder=5, label='Anomaly')
    ax1.axhline(y=18, color='blue', linestyle='--', alpha=0.5, label='Min Normal (18°C)')
    ax1.axhline(y=40, color='orange', linestyle='--', label='Max Normal (40°C)')
    ax1.axhline(y=50, color='red', linestyle='--', label='Critical (>50°C)')
    ax1.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
    ax1.set_title('Equipment Temperature', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power Consumption
    ax2.plot(range(len(power_consumptions)), power_consumptions, 'g-', linewidth=2, label='Power')
    if anomaly_indices:
        ax2.scatter([anomaly_indices[-1]] if anomaly_indices else [], 
                   [power_consumptions[anomaly_indices[-1]]] if anomaly_indices else [], 
                   color='red', s=100, zorder=5)
    ax2.axhline(y=2500, color='blue', linestyle='--', alpha=0.5, label='Min Normal (2500W)')
    ax2.axhline(y=3000, color='orange', linestyle='--', label='Max Normal (3000W)')
    ax2.axhline(y=3500, color='red', linestyle='--', label='Critical (>3500W)')
    ax2.set_ylabel('Power (Watts)', fontsize=10, fontweight='bold')
    ax2.set_title('Power Consumption', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signal Strength
    ax3.plot(range(len(signal_strengths)), signal_strengths, 'm-', linewidth=2, label='Signal')
    if anomaly_indices:
        ax3.scatter([anomaly_indices[-1]] if anomaly_indices else [], 
                   [signal_strengths[anomaly_indices[-1]]] if anomaly_indices else [], 
                   color='red', s=100, zorder=5)
    ax3.axhline(y=-75, color='blue', linestyle='--', alpha=0.5, label='Max Normal (-75dBm)')
    ax3.axhline(y=-95, color='orange', linestyle='--', label='Min Normal (-95dBm)')
    ax3.axhline(y=-105, color='red', linestyle='--', label='Critical (<-105dBm)')
    ax3.set_ylabel('Signal (dBm)', fontsize=10, fontweight='bold')
    ax3.set_title('Signal Strength', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Network Load
    ax4.plot(range(len(network_loads)), network_loads, 'c-', linewidth=2, label='Network Load')
    if anomaly_indices:
        ax4.scatter([anomaly_indices[-1]] if anomaly_indices else [], 
                   [network_loads[anomaly_indices[-1]]] if anomaly_indices else [], 
                   color='red', s=100, zorder=5)
    ax4.axhline(y=40, color='blue', linestyle='--', alpha=0.5, label='Min Normal (40%)')
    ax4.axhline(y=80, color='orange', linestyle='--', label='Max Normal (80%)')
    ax4.axhline(y=90, color='red', linestyle='--', label='Critical (>90%)')
    ax4.set_ylabel('Load (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Network Load', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (samples)', fontsize=9)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Prediction Confidence
    colors = ['red' if label != 'normal' else 'green' for label in anomaly_predictions]
    ax5.scatter(range(len(anomaly_confidences)), anomaly_confidences, c=colors, s=30, alpha=0.6)
    ax5.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, label='Confidence Threshold')
    ax5.set_ylabel('Prediction Confidence', fontsize=10, fontweight='bold')
    ax5.set_title('ML Model Confidence Score', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (samples)', fontsize=9)
    ax5.set_ylim([0, 1])
    ax5.legend(loc='lower left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Status Display
    ax6.axis('off')
    
    # Get model performance stats
    stats = detector.get_model_stats()
    
    # Status display
    status_text = f"SUPERVISED LEARNING STATUS\n"
    status_text += f"{'=' * 45}\n\n"
    status_text += f"Station ID: {base_station.station_id}\n"
    status_text += f"Time: {data['timestamp'].strftime('%H:%M:%S')}\n\n"
    status_text += f"Current Readings:\n"
    status_text += f"Temp: {data['temperature_C']:.1f}°C\n"
    status_text += f"Power: {data['power_consumption_W']:.0f}W\n"
    status_text += f"Signal: {data['signal_strength_dBm']:.1f}dBm\n"
    status_text += f"Load: {data['network_load_percent']:.1f}%\n\n"
    
    # ML Prediction Results
    status_text += f"ML Prediction:\n"
    status_text += f"Predicted: {predicted_label.upper()}\n"
    status_text += f"Actual: {data['anomaly_label'].upper()}\n"
    status_text += f"Confidence: {confidence*100:.1f}%\n"
    match_symbol = "✓" if predicted_label == data['anomaly_label'] else "✗"
    status_text += f"Match: {match_symbol}\n\n"
    
    status_text += f"Maintenance:\n"
    status_text += f"{message}\n\n"
    
    # Model Stats
    status_text += f"Model (v{stats['version']}): Random Forest\n"
    status_text += f"Accuracy: {stats['accuracy']:.1f}%\n"
    status_text += f"Predictions: {stats['total_predictions']}\n"
    status_text += f"Next Retrain: {stats['samples_until_retrain']} samples\n"
    
    # Email status
    if email_notifier:
        status_text += f"\nEmail Alerts: ENABLED ✉️\n"
    else:
        status_text += f"\nEmail Alerts: DISABLED\n"
    
    # Color code based on severity
    text_color = 'red' if severity_level == 'CRITICAL' else 'orange' if severity_level == 'WARNING' else 'green'
    
    ax6.text(0.05, 0.95, status_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color=text_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Print to console
    if is_anomaly:
        print(f"🚨 ANOMALY DETECTED: {predicted_label.upper()} (Confidence: {confidence*100:.1f}%) | Actual: {data['anomaly_label'].upper()}")


# Create animation
print("📊 Opening live dashboard...")
ani = animation.FuncAnimation(fig, update_plot, interval=2000, cache_frame_data=False)

plt.show()

print("\n✅ Monitoring session ended")
print(f"📊 Final Model Statistics:")
final_stats = detector.get_model_stats()
print(f"   Accuracy: {final_stats['accuracy']:.2f}%")
print(f"   Total Predictions: {final_stats['total_predictions']}")
print(f"💾 Model saved to: {detector.model_path}")
print(f"🔢 Model version: v{detector.model_version}")
