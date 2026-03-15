"""
Generate Labeled Training Dataset for 5G Base Station Anomaly Detection
========================================================================

Creates a CSV file with balanced labeled data:
- Normal operating conditions (90%)
- Various anomaly types (10%)
- All values respect threshold boundaries

Thresholds:
- Temperature: 18-40°C (normal)
- Power: 2500-3000W (normal)
- Signal: -95 to -75 dBm (normal)
- Load: 40-80% (normal)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
TOTAL_SAMPLES = 10000
ANOMALY_RATE = 0.10  # 10% anomalies
OUTPUT_FILE = "data/5g_training_data_labeled.csv"

# Normal operating thresholds
TEMP_MIN, TEMP_MAX = 18, 40
POWER_MIN, POWER_MAX = 2500, 3000
SIGNAL_MIN, SIGNAL_MAX = -95, -75  # Higher is better
LOAD_MIN, LOAD_MAX = 40, 80

# Anomaly types
ANOMALY_TYPES = [
    'overheating',
    'power_surge', 
    'signal_degradation',
    'network_overload',
    'cooling_failure',
    'equipment_failure'
]

def generate_normal_sample(timestamp):
    """Generate a normal operating sample within thresholds"""
    return {
        'timestamp': timestamp,
        'temperature_C': np.random.uniform(TEMP_MIN + 2, TEMP_MAX - 2),
        'power_consumption_W': np.random.uniform(POWER_MIN + 50, POWER_MAX - 50),
        'signal_strength_dBm': np.random.uniform(SIGNAL_MIN + 2, SIGNAL_MAX - 2),
        'network_load_percent': np.random.uniform(LOAD_MIN + 5, LOAD_MAX - 5),
        'anomaly_type': 'normal'
    }

def generate_anomaly_sample(timestamp, anomaly_type):
    """Generate an anomaly sample outside normal thresholds"""
    
    # Start with normal base values
    temp = np.random.uniform(TEMP_MIN + 2, TEMP_MAX - 2)
    power = np.random.uniform(POWER_MIN + 50, POWER_MAX - 50)
    signal = np.random.uniform(SIGNAL_MIN + 2, SIGNAL_MAX - 2)
    load = np.random.uniform(LOAD_MIN + 5, LOAD_MAX - 5)
    
    if anomaly_type == 'overheating':
        # Temperature exceeds maximum
        temp = np.random.uniform(TEMP_MAX + 3, TEMP_MAX + 25)
        power = np.random.uniform(POWER_MIN, POWER_MAX + 200)  # Slightly higher power
        
    elif anomaly_type == 'power_surge':
        # Power consumption exceeds maximum
        power = np.random.uniform(POWER_MAX + 200, POWER_MAX + 1000)
        temp = np.random.uniform(TEMP_MIN, TEMP_MAX + 5)  # Slightly higher temp
        
    elif anomaly_type == 'signal_degradation':
        # Signal strength below minimum (more negative = weaker)
        signal = np.random.uniform(-115, SIGNAL_MIN - 3)
        load = np.random.uniform(LOAD_MIN - 20, LOAD_MIN + 10)  # Lower load
        
    elif anomaly_type == 'network_overload':
        # Network load exceeds maximum
        load = np.random.uniform(LOAD_MAX + 5, 98)
        temp = np.random.uniform(TEMP_MIN, TEMP_MAX + 8)  # Higher temp due to load
        power = np.random.uniform(POWER_MIN, POWER_MAX + 300)  # Higher power
        
    elif anomaly_type == 'cooling_failure':
        # Severe overheating
        temp = np.random.uniform(TEMP_MAX + 15, TEMP_MAX + 30)
        power = np.random.uniform(POWER_MIN, POWER_MAX)
        
    elif anomaly_type == 'equipment_failure':
        # Multiple parameters out of range
        temp = np.random.uniform(TEMP_MIN - 10, TEMP_MIN + 2)  # Too cold
        power = np.random.uniform(POWER_MIN - 800, POWER_MIN - 100)  # Too low
        signal = np.random.uniform(-120, -105)  # Very weak
        load = np.random.uniform(5, 25)  # Very low load
    
    return {
        'timestamp': timestamp,
        'temperature_C': round(temp, 2),
        'power_consumption_W': round(power, 2),
        'signal_strength_dBm': round(signal, 2),
        'network_load_percent': round(np.clip(load, 0, 100), 2),
        'anomaly_type': anomaly_type
    }

def generate_dataset():
    """Generate complete labeled dataset"""
    
    print("=" * 70)
    print("GENERATING LABELED TRAINING DATASET")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   Total Samples: {TOTAL_SAMPLES:,}")
    print(f"   Anomaly Rate: {ANOMALY_RATE * 100:.1f}%")
    print(f"   Normal Samples: {int(TOTAL_SAMPLES * (1 - ANOMALY_RATE)):,}")
    print(f"   Anomaly Samples: {int(TOTAL_SAMPLES * ANOMALY_RATE):,}")
    print(f"\n🎯 Normal Operating Ranges:")
    print(f"   Temperature: {TEMP_MIN}°C - {TEMP_MAX}°C")
    print(f"   Power: {POWER_MIN}W - {POWER_MAX}W")
    print(f"   Signal: {SIGNAL_MIN}dBm - {SIGNAL_MAX}dBm")
    print(f"   Load: {LOAD_MIN}% - {LOAD_MAX}%")
    print(f"\n🚨 Anomaly Types: {len(ANOMALY_TYPES)}")
    for i, atype in enumerate(ANOMALY_TYPES, 1):
        print(f"   {i}. {atype}")
    
    data = []
    start_time = datetime.now()
    
    # Calculate samples per category
    normal_samples = int(TOTAL_SAMPLES * (1 - ANOMALY_RATE))
    anomaly_samples = TOTAL_SAMPLES - normal_samples
    samples_per_anomaly = anomaly_samples // len(ANOMALY_TYPES)
    
    print(f"\n⚙️  Generating samples...")
    
    # Generate normal samples
    print(f"   [1/{len(ANOMALY_TYPES) + 1}] Generating {normal_samples:,} normal samples...")
    for i in range(normal_samples):
        timestamp = start_time + timedelta(seconds=i * 2)
        data.append(generate_normal_sample(timestamp))
    
    # Generate anomaly samples (balanced across types)
    for idx, anomaly_type in enumerate(ANOMALY_TYPES):
        print(f"   [{idx + 2}/{len(ANOMALY_TYPES) + 1}] Generating {samples_per_anomaly:,} {anomaly_type} samples...")
        for i in range(samples_per_anomaly):
            timestamp = start_time + timedelta(seconds=(normal_samples + idx * samples_per_anomaly + i) * 2)
            data.append(generate_anomaly_sample(timestamp, anomaly_type))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    print(f"\n🔀 Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add row ID
    df.insert(0, 'id', range(1, len(df) + 1))
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"   📁 Saved to: {OUTPUT_FILE}")
    print(f"   📊 Total rows: {len(df):,}")
    print(f"   💾 File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
    
    # Show class distribution
    print(f"\n📊 Class Distribution:")
    class_counts = df['anomaly_type'].value_counts().sort_index()
    for anomaly_type, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {anomaly_type:20s}: {count:5,} ({percentage:5.2f}%)")
    
    # Show sample statistics
    print(f"\n📈 Feature Statistics:")
    print(f"\n   Temperature (°C):")
    print(f"      Min: {df['temperature_C'].min():.2f}")
    print(f"      Max: {df['temperature_C'].max():.2f}")
    print(f"      Mean: {df['temperature_C'].mean():.2f}")
    
    print(f"\n   Power (W):")
    print(f"      Min: {df['power_consumption_W'].min():.2f}")
    print(f"      Max: {df['power_consumption_W'].max():.2f}")
    print(f"      Mean: {df['power_consumption_W'].mean():.2f}")
    
    print(f"\n   Signal (dBm):")
    print(f"      Min: {df['signal_strength_dBm'].min():.2f}")
    print(f"      Max: {df['signal_strength_dBm'].max():.2f}")
    print(f"      Mean: {df['signal_strength_dBm'].mean():.2f}")
    
    print(f"\n   Load (%):")
    print(f"      Min: {df['network_load_percent'].min():.2f}")
    print(f"      Max: {df['network_load_percent'].max():.2f}")
    print(f"      Mean: {df['network_load_percent'].mean():.2f}")
    
    # Show first few samples
    print(f"\n📋 Sample Data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    print(f"\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    generate_dataset()
