"""
SUPERVISED REAL-TIME ANOMALY DETECTION DASHBOARD
================================================

Multi-class anomaly classification using Random Forest Classifier
- Classifies 7 types of anomalies (supervised learning)
- Real-time monitoring with live ML metrics
- Confusion matrix and performance tracking
- Auto-refresh every 2 seconds

Author: Industrial IoT Monitoring System
Version: 2.0 (Supervised Learning)
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Import email notifier
try:
    from email_notifier import EmailNotificationSystem
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    print("⚠️ Email notifier module not found. Email notifications disabled.")

warnings.filterwarnings('ignore')

# Ensure imports work regardless of where Streamlit is launched from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class FiveGBaseStation:
    """Simulates a real 5G telecom base station with labeled anomaly types"""
    
    def __init__(self, station_id="5G-BS-001"):
        self.station_id = station_id
        # Updated normal operating parameters
        self.normal_temp = 29.0      # Mid-point of 18-40°C
        self.normal_power = 2750     # Mid-point of 2500-3000W
        self.normal_signal = -85     # Mid-point of -75 to -95 dBm
        self.normal_load = 60        # Mid-point of 40-80%
        
        # Normal operating thresholds
        self.temp_min, self.temp_max = 18, 40
        self.power_min, self.power_max = 2500, 3000
        self.signal_min, self.signal_max = -95, -75
        self.load_min, self.load_max = 40, 80
        
        self.time_counter = 0
        
    def generate_sensor_data(self):
        """Generate sensor reading with random anomalies"""
        self.time_counter += 1
        
        # Start with normal values (smaller variation for tighter thresholds)
        temperature = self.normal_temp + random.uniform(-3, 3)
        power_consumption = self.normal_power + random.uniform(-100, 100)
        signal_strength = self.normal_signal + random.uniform(-3, 3)
        network_load = self.normal_load + random.uniform(-8, 8)
        
        # Randomly inject anomalies (5% chance to match training data)
        anomaly_label = 'normal'
        if random.random() < 0.10:
            anomaly_type = random.choice([
                'overheating', 'power_surge', 'signal_degradation',
                'network_overload', 'cooling_failure', 'equipment_failure'
            ])
            
            if anomaly_type == 'overheating':
                temperature = random.uniform(self.temp_max + 5, self.temp_max + 20)  # Above 40°C
                anomaly_label = 'overheating'
            elif anomaly_type == 'power_surge':
                power_consumption = random.uniform(self.power_max + 200, self.power_max + 800)  # Above 3000W
                anomaly_label = 'power_surge'
            elif anomaly_type == 'signal_degradation':
                signal_strength = random.uniform(-110, self.signal_min - 5)  # Below -95 dBm
                anomaly_label = 'signal_degradation'
            elif anomaly_type == 'network_overload':
                network_load = random.uniform(self.load_max + 5, 98)  # Above 80%
                temperature += random.uniform(5, 10)  # Temperature rises
                power_consumption += random.uniform(100, 300)  # Power increases
                anomaly_label = 'network_overload'
            elif anomaly_type == 'cooling_failure':
                temperature = random.uniform(self.temp_max + 10, self.temp_max + 25)  # Severe overheating
                anomaly_label = 'cooling_failure'
            elif anomaly_type == 'equipment_failure':
                temperature = random.uniform(self.temp_min - 10, self.temp_min)  # Below 18°C
                power_consumption = random.uniform(self.power_min - 500, self.power_min - 100)  # Below 2500W
                signal_strength = random.uniform(-115, -105)  # Very weak
                network_load = random.uniform(0, 20)  # Very low
                anomaly_label = 'equipment_failure'
        
        return {
            'timestamp': datetime.now(),
            'temperature_C': round(temperature, 2),
            'power_consumption_W': round(power_consumption, 2),
            'signal_strength_dBm': round(signal_strength, 2),
            'network_load_percent': round(network_load, 2),
            'anomaly_label': anomaly_label
        }


class SupervisedAnomalyDetector:
    """Supervised multi-class anomaly classifier using Random Forest"""
    
    def __init__(self, model_path="models/5g_supervised_model.pkl"):
        self.model_path = os.path.join(ROOT_DIR, model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None  # For models trained with LabelEncoder
        self.is_trained = False
        
        # Anomaly classes
        self.anomaly_classes = [
            'normal', 'overheating', 'power_surge', 'signal_degradation',
            'network_overload', 'cooling_failure', 'equipment_failure'
        ]
        
        self.feature_cols = ['temperature_C', 'power_consumption_W', 
                            'signal_strength_dBm', 'network_load_percent']
        
        # Prediction history for metrics
        self.predictions_history = []
        self.true_labels_history = []
        self.anomaly_scores_history = []
        
        # Detection threshold for anomaly score (0 = normal, higher = more anomalous)
        self.detection_threshold = 0.0  # Internal threshold for binary classification
        
        # Try to load existing model
        self.load_model()
        
        # If no model exists, train a new one
        if not self.is_trained:
            self._train_new_model()
    
    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            try:
                saved_data = joblib.load(self.model_path)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                
                # Load label encoder if it exists (for epoch-trained models)
                if 'label_encoder' in saved_data:
                    self.label_encoder = saved_data['label_encoder']
                    # Update anomaly classes from the encoder
                    self.anomaly_classes = list(self.label_encoder.classes_)
                    print(f"Loaded model with LabelEncoder - Classes: {self.anomaly_classes}")
                else:
                    self.label_encoder = None
                    print(f"Loaded model without LabelEncoder - Direct string labels")
                
                self.is_trained = True
                print(f"✅ Loaded supervised model from: {self.model_path}")
                
                # Print model info
                if 'best_val_acc' in saved_data:
                    print(f"   Best Validation Accuracy: {saved_data['best_val_acc']*100:.2f}%")
                if 'n_estimators' in saved_data:
                    print(f"   Number of Trees: {saved_data['n_estimators']}")
                    
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            print(f"Saved supervised model to: {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _train_new_model(self):
        """Train a new Random Forest classifier with synthetic data"""
        print("Training new supervised model with synthetic data...")
        
        # Generate training data
        training_data = []
        simulator = FiveGBaseStation("TRAINING")
        
        for _ in range(1000):
            data = simulator.generate_sensor_data()
            training_data.append(data)
        
        df = pd.DataFrame(training_data)
        X = df[self.feature_cols].values
        y = df['anomaly_label'].values
        
        # Train scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        print("Supervised model trained successfully!")
        print(f"   - Training samples: {len(training_data)}")
        print(f"   - Classes: {self.anomaly_classes}")
    
    def predict(self, data):
        """Predict anomaly class for new data and calculate anomaly score"""
        if not self.is_trained:
            return 'normal', 0.0, 0.0
        
        # Extract features
        X = np.array([[
            data['temperature_C'],
            data['power_consumption_W'],
            data['signal_strength_dBm'],
            data['network_load_percent']
        ]])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Convert prediction back to string label if model was trained with LabelEncoder
        if self.label_encoder is not None:
            # Model was trained with encoded labels (0, 1, 2, ...) - decode to strings
            prediction = self.label_encoder.inverse_transform([prediction])[0]
        
        # Convert prediction to string to avoid numpy type issues
        prediction = str(prediction)
        
        confidence = max(probabilities) * 100
        
        # Calculate anomaly score (probability of being an anomaly, not normal)
        # Find index of 'normal' class
        if self.label_encoder is not None:
            # For encoded models, find 'normal' in the label encoder classes
            normal_idx = list(self.label_encoder.classes_).index('normal') if 'normal' in self.label_encoder.classes_ else 0
        else:
            # For direct string models
            normal_idx = list(self.model.classes_).index('normal') if 'normal' in self.model.classes_ else 0
        
        # Anomaly score is 1 - probability of normal
        anomaly_score = 1.0 - probabilities[normal_idx]
        
        # Store prediction for metrics (store as string)
        if 'anomaly_label' in data:
            self.predictions_history.append(prediction)
            self.true_labels_history.append(str(data['anomaly_label']))
            self.anomaly_scores_history.append(anomaly_score)
            
            # Keep last 3000 predictions for metrics calculation
            if len(self.predictions_history) > 3000:
                self.predictions_history.pop(0)
                self.true_labels_history.pop(0)
                self.anomaly_scores_history.pop(0)
        
        return prediction, confidence, anomaly_score
    
    def get_classification_metrics(self):
        """Get detailed classification metrics from multi-class confusion matrix"""
        if len(self.predictions_history) < 10:
            return None
        
        accuracy = accuracy_score(self.true_labels_history, self.predictions_history)
        
        # Get unique labels that actually appeared
        unique_labels = sorted(list(set(self.true_labels_history + self.predictions_history)))
        
        # Calculate multi-class metrics using sklearn
        if len(unique_labels) >= 2:
            report = classification_report(
                self.true_labels_history,
                self.predictions_history,
                labels=unique_labels,
                target_names=unique_labels,
                output_dict=True,
                zero_division=0
            )
        else:
            report = None
        
        # Multi-class confusion matrix (7x7)
        cm = confusion_matrix(
            self.true_labels_history,
            self.predictions_history,
            labels=self.anomaly_classes
        )
        
        # Calculate Precision, Recall, and F1-Score manually from confusion matrix
        # For each class i:
        # Precision_i = TP_i / (TP_i + FP_i) = cm[i,i] / sum(cm[:,i])
        # Recall_i = TP_i / (TP_i + FN_i) = cm[i,i] / sum(cm[i,:])
        
        num_classes = len(self.anomaly_classes)
        class_precisions = []
        class_recalls = []
        class_f1_scores = []
        class_supports = []
        
        for i in range(num_classes):
            # True Positives for class i
            tp_i = cm[i, i]
            
            # False Positives for class i (predicted as i but actually other classes)
            fp_i = np.sum(cm[:, i]) - tp_i
            
            # False Negatives for class i (actually i but predicted as other classes)
            fn_i = np.sum(cm[i, :]) - tp_i
            
            # Support (actual count of this class)
            support_i = np.sum(cm[i, :])
            class_supports.append(support_i)
            
            # Precision for class i
            if (tp_i + fp_i) > 0:
                precision_i = tp_i / (tp_i + fp_i)
            else:
                precision_i = 0.0
            class_precisions.append(precision_i)
            
            # Recall for class i
            if (tp_i + fn_i) > 0:
                recall_i = tp_i / (tp_i + fn_i)
            else:
                recall_i = 0.0
            class_recalls.append(recall_i)
            
            # F1-Score for class i
            if (precision_i + recall_i) > 0:
                f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
            else:
                f1_i = 0.0
            class_f1_scores.append(f1_i)
        
        # Calculate weighted averages (weighted by support)
        total_support = sum(class_supports)
        if total_support > 0:
            multi_class_precision = sum(p * s for p, s in zip(class_precisions, class_supports)) / total_support * 100
            multi_class_recall = sum(r * s for r, s in zip(class_recalls, class_supports)) / total_support * 100
            multi_class_f1 = sum(f * s for f, s in zip(class_f1_scores, class_supports)) / total_support * 100
        else:
            multi_class_precision = 0.0
            multi_class_recall = 0.0
            multi_class_f1 = 0.0
        
        return {
            'accuracy': accuracy * 100,
            'precision': multi_class_precision,
            'recall': multi_class_recall,
            'f1_score': multi_class_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'total_predictions': len(self.predictions_history),
            'class_precisions': class_precisions,
            'class_recalls': class_recalls,
            'class_f1_scores': class_f1_scores,
            'class_supports': class_supports
        }


def create_5g_plots(df):
    """Create sensor reading plots with X markers for anomalies on relevant parameters only"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Power Consumption', 
                       'Signal Strength', 'Network Load'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Define which parameters are affected by each anomaly type
    anomaly_parameter_mapping = {
        'normal': [],
        'overheating': ['temperature'],  # Only temperature affected
        'power_surge': ['power'],  # Only power affected
        'signal_degradation': ['signal'],  # Only signal affected
        'network_overload': ['network', 'power'],  # Network load + power affected
        'cooling_failure': ['temperature', 'power'],  # Temperature + power affected
        'equipment_failure': ['temperature', 'power', 'signal', 'network']  # All parameters affected
    }
    
    # Create intelligent color/symbol arrays for each parameter
    def get_marker_style(anomaly_label, parameter_name):
        """Determine if this parameter should be marked as anomaly"""
        label_str = str(anomaly_label).lower()
        affected_params = anomaly_parameter_mapping.get(label_str, [])
        
        if label_str == 'normal':
            return 'green', 'circle'
        elif parameter_name in affected_params:
            return 'red', 'x'  # Mark as anomaly
        else:
            return 'green', 'circle'  # Keep normal (not affected by this anomaly)
    
    # Generate marker styles for each parameter
    temp_colors = [get_marker_style(label, 'temperature')[0] for label in df['actual_label']]
    temp_symbols = [get_marker_style(label, 'temperature')[1] for label in df['actual_label']]
    
    power_colors = [get_marker_style(label, 'power')[0] for label in df['actual_label']]
    power_symbols = [get_marker_style(label, 'power')[1] for label in df['actual_label']]
    
    signal_colors = [get_marker_style(label, 'signal')[0] for label in df['actual_label']]
    signal_symbols = [get_marker_style(label, 'signal')[1] for label in df['actual_label']]
    
    network_colors = [get_marker_style(label, 'network')[0] for label in df['actual_label']]
    network_symbols = [get_marker_style(label, 'network')[1] for label in df['actual_label']]
    
    # Temperature (only mark if temperature-related anomaly)
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['temperature_C'],
                  mode='lines+markers', name='Temperature',
                  marker=dict(color=temp_colors, size=10, symbol=temp_symbols),
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    
    # Power (only mark if power-related anomaly)
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['power_consumption_W'],
                  mode='lines+markers', name='Power',
                  marker=dict(color=power_colors, size=10, symbol=power_symbols),
                  line=dict(color='orange', width=2)),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Watts", row=1, col=2)
    
    # Signal (only mark if signal-related anomaly)
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['signal_strength_dBm'],
                  mode='lines+markers', name='Signal',
                  marker=dict(color=signal_colors, size=10, symbol=signal_symbols),
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.update_yaxes(title_text="dBm", row=2, col=1)
    
    # Network Load (only mark if network-related anomaly)
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['network_load_percent'],
                  mode='lines+markers', name='Network Load',
                  marker=dict(color=network_colors, size=10, symbol=network_symbols),
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )
    fig.update_yaxes(title_text="%", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Real-Time Sensor Readings (Smart Marking: Red X only on affected parameters)"
    )
    
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def safe_plotly_chart(fig):
    """Render Plotly chart with backward compatibility"""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def create_anomaly_timeline(df):
    """Create anomaly detection timeline showing anomaly scores over time"""
    if 'anomaly_score' not in df.columns or len(df) == 0:
        return
    
    # Create colors based on actual label (ground truth)
    colors = ['green' if label == 'normal' else 'red' for label in df['actual_label']]
    symbols = ['circle' if label == 'normal' else 'x' for label in df['actual_label']]
    
    fig = go.Figure()
    
    # Add anomaly score line
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['anomaly_score'],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='teal', width=2),
        marker=dict(color=colors, size=8, symbol=symbols),
        hovertemplate='<b>Sample %{x}</b><br>' +
                      'Anomaly Score: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Real-time Anomaly Detection Scores (Static Model)",
        xaxis_title="Time (samples)",
        yaxis_title="Anomaly Score",
        height=400,
        showlegend=True,
        hovermode='closest',
        yaxis=dict(range=[-0.1, 1.1])
    )
    
    safe_plotly_chart(fig)


def main():
    st.set_page_config(
        page_title="Supervised Real-Time Anomaly Detection",
        page_icon="🗼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🗼 Supervised Real-Time 5G Anomaly Detection Dashboard")
    st.markdown("**Multi-Class Anomaly Classification with Random Forest (Supervised Learning)**")
    
    # Add informational note about intelligent parameter marking
    st.info("ℹ️ **Smart Anomaly Marking:** Red X markers appear only on parameters affected by each anomaly type. " +
            "For example: Overheating 🔥 → only Temperature marked | Equipment Failure 🔧 → all parameters marked.")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (2 sec)", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔃 Manual Refresh"):
        st.rerun()
    
    # Reset metrics button
    if st.sidebar.button("🗑️ Reset All Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ All data, metrics, and anomaly counter reset!")
        time.sleep(0.5)
        st.rerun()
    
    # Test email button
    st.sidebar.markdown("---")
    if st.sidebar.button("📧 Send Test Email"):
        if EMAIL_AVAILABLE and 'email_notifier' in st.session_state and st.session_state['email_notifier']:
            try:
                notifier = st.session_state['email_notifier']
                st.sidebar.info("📤 Sending test email...")
                success = notifier.send_test_email()
                if success:
                    st.sidebar.success("✅ Test email sent! Check your inbox.")
                else:
                    st.sidebar.error("❌ Test email failed. Check console output.")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.sidebar.error("❌ Email notifier not available")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dashboard Features:**")
    st.sidebar.markdown("• Supervised learning (Random Forest)")
    st.sidebar.markdown("• Multi-class classification (7 types)")
    st.sidebar.markdown("• Real-time performance metrics")
    st.sidebar.markdown("• Confusion matrix visualization")
    st.sidebar.markdown("• Auto-refresh every 2 seconds")
    st.sidebar.markdown("• **📧 Email notifications**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📧 Email Alert Thresholds:**")
    st.sidebar.markdown("• **5 anomalies**: INFO email")
    st.sidebar.markdown("• **10 anomalies**: WARNING email")
    st.sidebar.markdown("• **20 anomalies**: CRITICAL email")
    st.sidebar.markdown("• **20+ anomalies**: CRITICAL every 5")
    
    # Display current anomaly count
    anomaly_count = st.session_state.get('anomaly_count', 0)
    st.sidebar.markdown("---")
    if anomaly_count >= 20:
        st.sidebar.error(f"🚨 **Anomaly Count: {anomaly_count}**")
    elif anomaly_count >= 10:
        st.sidebar.warning(f"⚠️ **Anomaly Count: {anomaly_count}**")
    elif anomaly_count >= 5:
        st.sidebar.info(f"ℹ️ **Anomaly Count: {anomaly_count}**")
    else:
        st.sidebar.success(f"✅ **Anomaly Count: {anomaly_count}**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Anomaly Classes:**")
    st.sidebar.markdown("1. ✅ Normal")
    st.sidebar.markdown("2. 🔥 Overheating")
    st.sidebar.markdown("3. ⚡ Power Surge")
    st.sidebar.markdown("4. 📡 Signal Degradation")
    st.sidebar.markdown("5. 📊 Network Overload")
    st.sidebar.markdown("6. ❄️ Cooling Failure")
    st.sidebar.markdown("7. ⚙️ Equipment Failure")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 Parameter Dependencies:**")
    st.sidebar.markdown("🔥 **Overheating** → 🌡️ Temp")
    st.sidebar.markdown("⚡ **Power Surge** → ⚡ Power")
    st.sidebar.markdown("📡 **Signal Degradation** → 📶 Signal")
    st.sidebar.markdown("📊 **Network Overload** → 📊 Network + ⚡ Power")
    st.sidebar.markdown("❄️ **Cooling Failure** → 🌡️ Temp + ⚡ Power")
    st.sidebar.markdown("⚙️ **Equipment Failure** → 🌡️🔋📶📊 All")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📏 Normal Operating Thresholds:**")
    st.sidebar.markdown("**🌡️ Temperature:**")
    st.sidebar.markdown("• Min: 18.0°C")
    st.sidebar.markdown("• Max: 40.0°C")
    st.sidebar.markdown("")
    st.sidebar.markdown("**⚡ Power Consumption:**")
    st.sidebar.markdown("• Min: 2500 W")
    st.sidebar.markdown("• Max: 3000 W")
    st.sidebar.markdown("")
    st.sidebar.markdown("**📡 Signal Strength:**")
    st.sidebar.markdown("• Min: -95.0 dBm (weaker)")
    st.sidebar.markdown("• Max: -75.0 dBm (stronger)")
    st.sidebar.markdown("")
    st.sidebar.markdown("**📊 Network Load:**")
    st.sidebar.markdown("• Min: 40.0%")
    st.sidebar.markdown("• Max: 80.0%")
    
    # Initialize session state
    if 'simulator_supervised' not in st.session_state:
        st.session_state['simulator_supervised'] = FiveGBaseStation("5G-SUPERVISED")
    
    if 'detector_supervised' not in st.session_state:
        st.session_state['detector_supervised'] = SupervisedAnomalyDetector()
        
        # Check if model was loaded or newly trained
        detector_temp = st.session_state['detector_supervised']
        if os.path.exists(detector_temp.model_path):
            st.sidebar.success("✅ Model loaded from file")
        else:
            st.sidebar.warning("⚠️ No model found - training new model with 1000 samples. For better accuracy (97-99%), run: `python scripts/train_model_with_epochs.py`")
    
    if 'data_history_supervised' not in st.session_state:
        st.session_state['data_history_supervised'] = []
    
    # Initialize email notifier
    if 'email_notifier' not in st.session_state and EMAIL_AVAILABLE:
        try:
            st.session_state['email_notifier'] = EmailNotificationSystem()
            st.sidebar.success("✅ Email notifications enabled")
        except Exception as e:
            st.sidebar.error(f"⚠️ Email notifier failed: {str(e)}")
            st.session_state['email_notifier'] = None
    elif not EMAIL_AVAILABLE:
        st.session_state['email_notifier'] = None
        st.sidebar.warning("⚠️ Email module not available")
    
    # Initialize anomaly counter
    if 'anomaly_count' not in st.session_state:
        st.session_state['anomaly_count'] = 0
    
    if 'last_email_at_count' not in st.session_state:
        st.session_state['last_email_at_count'] = 0
    
    # Get current instances
    simulator = st.session_state['simulator_supervised']
    detector = st.session_state['detector_supervised']
    data_history = st.session_state['data_history_supervised']
    email_notifier = st.session_state.get('email_notifier', None)
    
    # Generate new data point
    new_data = simulator.generate_sensor_data()
    
    # Predict anomaly class and get anomaly score
    predicted_label, confidence, anomaly_score = detector.predict(new_data)
    new_data['predicted_label'] = predicted_label
    new_data['confidence'] = confidence
    new_data['anomaly_score'] = anomaly_score
    new_data['actual_label'] = new_data['anomaly_label']  # Store ground truth
    
    # Track anomaly count and send emails based on thresholds
    if str(predicted_label) != 'normal':
        st.session_state['anomaly_count'] += 1
        anomaly_count = st.session_state['anomaly_count']
        last_email_at = st.session_state['last_email_at_count']
        
        # Determine if we should send an email
        should_send_email = False
        severity = None
        
        if anomaly_count == 5:
            should_send_email = True
            severity = 'INFO'
        elif anomaly_count == 10:
            should_send_email = True
            severity = 'WARNING'
        elif anomaly_count == 20:
            should_send_email = True
            severity = 'CRITICAL'
        elif anomaly_count > 20 and (anomaly_count - last_email_at) >= 5:
            should_send_email = True
            severity = 'CRITICAL'
        
        # Send email notification with detailed feedback
        if should_send_email:
            if email_notifier is None:
                st.sidebar.warning(f"⚠️ Email notifier not initialized (Anomaly #{anomaly_count})")
            else:
                try:
                    # Get model stats
                    metrics = detector.get_classification_metrics()
                    if metrics:
                        model_stats = {
                            'version': 2,
                            'accuracy': metrics['accuracy'] * 100,
                            'precision': metrics['precision'] * 100,
                            'recall': metrics['recall'] * 100,
                            'total_predictions': len(detector.predictions_history)
                        }
                    else:
                        model_stats = {
                            'version': 2,
                            'accuracy': 0,
                            'precision': 0,
                            'recall': 0,
                            'total_predictions': 0
                        }
                    
                    # Create maintenance message based on severity
                    if severity == 'INFO':
                        maintenance_msg = f"📊 Anomaly Alert: {anomaly_count} anomalies detected. Monitor the system closely for any developing issues."
                    elif severity == 'WARNING':
                        maintenance_msg = f"⚠️ Warning: {anomaly_count} anomalies detected. Schedule preventive maintenance soon to avoid critical failures."
                    elif severity == 'CRITICAL':
                        maintenance_msg = f"🚨 Critical: {anomaly_count} anomalies detected! Immediate inspection and maintenance required. System may experience degraded performance."
                    
                    st.sidebar.info(f"📤 Attempting to send {severity} email (Anomaly #{anomaly_count})...")
                    
                    # Send email
                    email_sent = email_notifier.send_email_notification(
                        severity=severity,
                        station_id=simulator.station_id,
                        timestamp=new_data['timestamp'],
                        anomaly_data={
                            'anomaly_type': str(predicted_label),
                            'temperature_C': new_data['temperature_C'],
                            'power_consumption_W': new_data['power_consumption_W'],
                            'signal_strength_dBm': new_data['signal_strength_dBm'],
                            'network_load_percent': new_data['network_load_percent']
                        },
                        maintenance_message=maintenance_msg,
                        model_stats=model_stats
                    )
                    
                    if email_sent:
                        st.session_state['last_email_at_count'] = anomaly_count
                        st.sidebar.success(f"✅ {severity} email sent successfully! (Anomaly #{anomaly_count})")
                        st.balloons()  # Celebration effect
                    else:
                        st.sidebar.warning(f"⏳ Email blocked by cooldown or failed (Anomaly #{anomaly_count})")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Email error at anomaly #{anomaly_count}: {str(e)}")
                    import traceback
                    st.sidebar.text(traceback.format_exc())
    
    # Add to history
    data_history.append(new_data)
    
    # Keep only recent data (last 100 points)
    if len(data_history) > 100:
        data_history.pop(0)
    
    st.session_state['data_history_supervised'] = data_history
    
    # CURRENT SENSOR READINGS
    st.subheader("📊 Current Sensor Readings")
    
    # Display email notification status banner
    anomaly_count = st.session_state.get('anomaly_count', 0)
    if anomaly_count > 0:
        col_banner1, col_banner2 = st.columns([3, 1])
        with col_banner1:
            if anomaly_count >= 20:
                st.error(f"🚨 **CRITICAL ALERT**: {anomaly_count} anomalies detected! Email notifications active.")
            elif anomaly_count >= 10:
                st.warning(f"⚠️ **WARNING**: {anomaly_count} anomalies detected. Monitoring closely.")
            elif anomaly_count >= 5:
                st.info(f"ℹ️ **NOTICE**: {anomaly_count} anomalies detected. System under observation.")
        with col_banner2:
            if email_notifier:
                st.success("📧 Email: ON")
            else:
                st.warning("📧 Email: OFF")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{new_data['temperature_C']}°C")
    with col2:
        st.metric("⚡ Power", f"{new_data['power_consumption_W']}W")
    with col3:
        st.metric("📡 Signal", f"{new_data['signal_strength_dBm']}dBm")
    with col4:
        st.metric("📊 Network Load", f"{new_data['network_load_percent']}%")
    
    # Current prediction
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    # Convert predicted_label to string (in case it's numpy int)
    predicted_label_str = str(predicted_label)
    actual_label = str(new_data.get('anomaly_label', 'unknown'))
    
    with col1:
        if predicted_label_str == 'normal':
            st.success(f"✅ **Prediction:** {predicted_label_str.upper()} (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"⚠️ **Prediction:** {predicted_label_str.upper()} (Confidence: {confidence:.1f}%)")
    
    with col2:
        if actual_label == predicted_label_str:
            st.info(f"✓ **Ground Truth:** {actual_label} (Correct)")
        else:
            st.warning(f"✗ **Ground Truth:** {actual_label} (Mismatch)")
    
    # REAL-TIME PLOTS
    st.markdown("---")
    
    if len(data_history) > 1:
        df = pd.DataFrame(data_history)
        create_5g_plots(df)
    
    # ANOMALY DETECTION TIMELINE
    st.markdown("---")
    st.subheader("🎯 Anomaly Detection Timeline")
    
    if len(data_history) > 1:
        df = pd.DataFrame(data_history)
        create_anomaly_timeline(df)
    
    # ML PERFORMANCE METRICS
    st.markdown("---")
    st.subheader("🤖 Machine Learning Performance Metrics (Static Model)")
    
    metrics = detector.get_classification_metrics()
    
    if metrics:
        # Metrics overview
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.1f}%")
        with col2:
            st.metric("🔍 Precision", f"{metrics['precision']:.1f}%")
        with col3:
            st.metric("📊 Recall", f"{metrics['recall']:.1f}%")
        with col4:
            st.metric("⚖️ F1-Score", f"{metrics['f1_score']:.1f}%")
        with col5:
            st.metric("📈 Total Predictions", metrics['total_predictions'])
        
        # Multi-Class Confusion Matrix and Metrics
        st.markdown("---")
        
        # Single column for multi-class confusion matrix
        st.subheader("🎯 Multi-Class Confusion Matrix (All 7 Anomaly Types)")
        
        cm_multi = metrics['confusion_matrix']
        fig_multi = px.imshow(
            cm_multi,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=detector.anomaly_classes,
            y=detector.anomaly_classes,
            color_continuous_scale='RdYlGn_r',
            text_auto=True,
            title="Multi-Class Confusion Matrix"
        )
        fig_multi.update_traces(textfont_size=14, texttemplate='%{z}')
        fig_multi.update_layout(
            height=650,
            font=dict(size=12),
            xaxis=dict(tickangle=45, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11))
        )
        safe_plotly_chart(fig_multi)
        
        # Metrics breakdown in columns
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Multi-Class Metrics Breakdown")
            
            st.info("**Metrics calculated from Multi-Class Confusion Matrix (7 classes)**")
            
            # Show the formulas with actual calculation using LaTeX
            st.markdown("### 📐 Calculation Formulas")
            st.markdown("")
            
            st.markdown("**For each class $i$:**")
            st.latex(r"\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}")
            st.latex(r"\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}")
            st.latex(r"\text{F1}_i = \frac{2 \times \text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}")
            
            st.markdown("")
            st.markdown("**Weighted Average (by class support):**")
            st.latex(r"\text{Precision} = \frac{\sum (\text{Precision}_i \times \text{Support}_i)}{\text{Total}}")
            st.latex(r"\text{Recall} = \frac{\sum (\text{Recall}_i \times \text{Support}_i)}{\text{Total}}")
            st.latex(r"\text{F1-Score} = \frac{\sum (\text{F1}_i \times \text{Support}_i)}{\text{Total}}")
            
            st.markdown("")
            st.success(f"**✅ Current Weighted Averages:**\n\n"
                      f"• Precision = **{metrics['precision']:.2f}%**\n\n"
                      f"• Recall = **{metrics['recall']:.2f}%**\n\n"
                      f"• F1-Score = **{metrics['f1_score']:.2f}%**")
        
        with col2:
            st.subheader("🔍 Per-Class Performance")
            
            # Show per-class breakdown in a cleaner format
            st.markdown("**Metrics for each anomaly class:**")
            st.markdown("")
            
            for i, class_name in enumerate(detector.anomaly_classes):
                if metrics['class_supports'][i] > 0:
                    # Create a nice box for each class
                    with st.container():
                        st.markdown(f"**🔹 {class_name.replace('_', ' ').upper()}**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Precision", f"{metrics['class_precisions'][i]*100:.1f}%", label_visibility="visible")
                        with col_b:
                            st.metric("Recall", f"{metrics['class_recalls'][i]*100:.1f}%", label_visibility="visible")
                        with col_c:
                            st.metric("F1-Score", f"{metrics['class_f1_scores'][i]*100:.1f}%", label_visibility="visible")
                        st.caption(f"📊 Support: {metrics['class_supports'][i]} samples")
                        st.markdown("---")
        
        # Classification report table (if available)
        if metrics['classification_report']:
            st.markdown("---")
            st.subheader("📈 Detailed Per-Class Performance Table")
            
            report = metrics['classification_report']
            report_data = []
            
            for class_name in detector.anomaly_classes:
                if class_name in report:
                    report_data.append({
                        'Class': class_name.capitalize(),
                        'Precision': f"{report[class_name]['precision']*100:.1f}%",
                        'Recall': f"{report[class_name]['recall']*100:.1f}%",
                        'F1-Score': f"{report[class_name]['f1-score']*100:.1f}%",
                        'Support': int(report[class_name]['support'])
                    })
            
            if report_data:
                df_report = pd.DataFrame(report_data)
                st.dataframe(df_report, use_container_width=True)
        
        # Debug section - Show prediction details
        with st.expander("🔍 Debug: View Recent Predictions"):
            recent_count = min(20, len(detector.predictions_history))
            debug_data = {
                'Actual': detector.true_labels_history[-recent_count:],
                'Predicted': detector.predictions_history[-recent_count:],
                'Match': ['✅' if a == p else '❌' for a, p in zip(
                    detector.true_labels_history[-recent_count:],
                    detector.predictions_history[-recent_count:]
                )]
            }
            st.dataframe(pd.DataFrame(debug_data), use_container_width=True)
    
    else:
        st.info("📊 Collecting predictions... Metrics will appear after 10+ samples.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
