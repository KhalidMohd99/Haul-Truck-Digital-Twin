"""
LSTM Anomaly Detection & RUL Prediction for Haul Truck Fleet
Mansourah-Massarah Gold Mine, Saudi Arabia

This template provides complete code for training an LSTM model
on the generated telemetry data for predictive maintenance.

Author: Industrial IoT Simulation
Dataset: mansourah_haul_truck_telemetry.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': 'mansourah_haul_truck_telemetry.csv',
    'sequence_length': 80,  # 40 minutes (80 * 30 seconds)
    'stride': 15,  # Overlapping windows
    'train_days': 5,  # Days 1-5 for training
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'lstm_units': [128, 64],  # 2-layer LSTM
    'dropout_rate': 0.3,
    'random_seed': 42
}

np.random.seed(CONFIG['random_seed'])
tf.random.set_seed(CONFIG['random_seed'])

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(filepath):
    """Load telemetry data and prepare for LSTM training"""
    
    print("Loading telemetry data...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Sort by truck and timestamp
    df = df.sort_values(['truck_id', 'timestamp']).reset_index(drop=True)
    
    # Create time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # One-hot encode operational state
    state_dummies = pd.get_dummies(df['operational_state'], prefix='state')
    df = pd.concat([df, state_dummies], axis=1)
    
    return df

def create_derived_features(df, window=10):
    """Create rolling statistics and derived features"""
    
    print("Creating derived features...")
    
    for truck_id in df['truck_id'].unique():
        mask = df['truck_id'] == truck_id
        
        # Rolling statistics for vibration
        df.loc[mask, 'vibration_rolling_mean'] = df.loc[mask, 'vibration_g'].rolling(
            window=window, min_periods=1).mean()
        df.loc[mask, 'vibration_rolling_std'] = df.loc[mask, 'vibration_g'].rolling(
            window=window, min_periods=1).std().fillna(0)
        
        # Temperature rate of change
        df.loc[mask, 'temp_rate_change'] = df.loc[mask, 'engine_temp_celsius'].diff().fillna(0)
        
        # Speed acceleration
        df.loc[mask, 'acceleration'] = df.loc[mask, 'speed_kph'].diff().fillna(0)
    
    return df

def split_train_val_test(df, train_days=5):
    """Split data into train, validation, and test sets"""
    
    print("\nSplitting data into train/val/test sets...")
    
    # Calculate day number
    df['day'] = (df['timestamp'] - df['timestamp'].min()).dt.days + 1
    
    # Training: Days 1-5, all trucks
    train_df = df[df['day'] <= train_days].copy()
    
    # Validation: Day 6+, healthy trucks only (exclude Truck 7)
    val_df = df[(df['day'] > train_days) & (df['truck_id'] != 7)].copy()
    
    # Test: Day 6+, all trucks (includes Truck 7 failure)
    test_df = df[df['day'] > train_days].copy()
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Test samples (Truck 7): {len(test_df[test_df['truck_id']==7]):,}")
    
    return train_df, val_df, test_df

# ============================================================================
# SEQUENCE GENERATION
# ============================================================================

def create_sequences(df, sequence_length, stride, feature_cols, target_col='rul_hours'):
    """Create sequences for LSTM training"""
    
    sequences = []
    targets = []
    truck_ids = []
    timestamps = []
    
    for truck_id in df['truck_id'].unique():
        truck_data = df[df['truck_id'] == truck_id].reset_index(drop=True)
        
        # Extract features and target
        features = truck_data[feature_cols].values
        target = truck_data[target_col].values
        truck_time = truck_data['timestamp'].values
        
        # Create sequences with stride
        for i in range(0, len(features) - sequence_length + 1, stride):
            sequences.append(features[i:i+sequence_length])
            targets.append(target[i+sequence_length-1])  # Predict RUL at end of sequence
            truck_ids.append(truck_id)
            timestamps.append(truck_time[i+sequence_length-1])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    truck_ids = np.array(truck_ids)
    
    print(f"  Created {len(sequences):,} sequences of shape {sequences.shape}")
    
    return sequences, targets, truck_ids, timestamps

# ============================================================================
# LSTM MODEL ARCHITECTURE
# ============================================================================

def build_lstm_model(input_shape, lstm_units=[128, 64], dropout_rate=0.3):
    """Build LSTM model for RUL prediction"""
    
    model = Sequential([
        Input(shape=input_shape),
        
        # First LSTM layer with return sequences
        LSTM(lstm_units[0], return_sequences=True),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(lstm_units[1], return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(16, activation='relu'),
        
        # Output layer for RUL prediction
        Dense(1, activation='linear')  # Regression output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def build_bidirectional_lstm(input_shape, lstm_units=[128, 64], dropout_rate=0.3):
    """Build Bidirectional LSTM for improved performance"""
    
    model = Sequential([
        Input(shape=input_shape),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(lstm_units[0], return_sequences=True)),
        Dropout(dropout_rate),
        
        Bidirectional(LSTM(lstm_units[1], return_sequences=False)),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_lstm_model():
    """Complete training pipeline"""
    
    # 1. Load and prepare data
    df = load_and_prepare_data(CONFIG['data_path'])
    df = create_derived_features(df)
    
    # 2. Define feature columns
    feature_cols = [
        'engine_temp_celsius', 'vibration_g', 'tire_pressure_psi',
        'speed_kph', 'fuel_consumption_lph', 'ambient_temp_celsius',
        'time_in_state_sec', 'hour_sin', 'hour_cos',
        'vibration_rolling_mean', 'vibration_rolling_std',
        'temp_rate_change', 'acceleration',
        'state_DUMPING', 'state_EMPTY_RETURN', 'state_LOADED_HAUL',
        'state_LOADING', 'state_QUEUE'
    ]
    
    # 3. Split data
    train_df, val_df, test_df = split_train_val_test(df, CONFIG['train_days'])
    
    # 4. Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 5. Create sequences
    print("\nCreating sequences...")
    print("Training sequences:")
    X_train, y_train, train_ids, _ = create_sequences(
        train_df, CONFIG['sequence_length'], CONFIG['stride'], 
        feature_cols, 'rul_hours'
    )
    
    print("Validation sequences:")
    X_val, y_val, val_ids, _ = create_sequences(
        val_df, CONFIG['sequence_length'], CONFIG['stride'],
        feature_cols, 'rul_hours'
    )
    
    print("Test sequences:")
    X_test, y_test, test_ids, test_timestamps = create_sequences(
        test_df, CONFIG['sequence_length'], CONFIG['stride'],
        feature_cols, 'rul_hours'
    )
    
    # 6. Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(
        input_shape=(CONFIG['sequence_length'], len(feature_cols)),
        lstm_units=CONFIG['lstm_units'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    print(model.summary())
    
    # 7. Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 8. Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("OVERALL TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"MAE: {mae:.2f} hours")
    print(f"R²: {r2:.4f}")
    
    # Truck 7 specific metrics
    truck7_mask = test_ids == 7
    if np.any(truck7_mask):
        y_test_truck7 = y_test[truck7_mask]
        y_pred_truck7 = y_pred[truck7_mask]
        
        rmse_truck7 = np.sqrt(mean_squared_error(y_test_truck7, y_pred_truck7))
        mae_truck7 = mean_absolute_error(y_test_truck7, y_pred_truck7)
        
        print(f"\n{'='*60}")
        print("TRUCK 7 (FAILING BEARING) PERFORMANCE")
        print(f"{'='*60}")
        print(f"RMSE: {rmse_truck7:.2f} hours")
        print(f"MAE: {mae_truck7:.2f} hours")
        print(f"Samples: {len(y_test_truck7):,}")
    
    # Healthy trucks metrics
    healthy_mask = test_ids != 7
    if np.any(healthy_mask):
        y_test_healthy = y_test[healthy_mask]
        y_pred_healthy = y_pred[healthy_mask]
        
        rmse_healthy = np.sqrt(mean_squared_error(y_test_healthy, y_pred_healthy))
        mae_healthy = mean_absolute_error(y_test_healthy, y_pred_healthy)
        
        print(f"\n{'='*60}")
        print("HEALTHY TRUCKS PERFORMANCE")
        print(f"{'='*60}")
        print(f"RMSE: {rmse_healthy:.2f} hours")
        print(f"MAE: {mae_healthy:.2f} hours")
        print(f"Samples: {len(y_test_healthy):,}")
    
    # 10. Save results
    results = {
        'model': model,
        'history': history.history,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_ids': test_ids,
        'test_timestamps': test_timestamps,
        'config': CONFIG
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation loss"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[0].set_title('Model Loss Over Epochs', fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # MAE
    axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('MAE (hours)', fontweight='bold')
    axes[1].set_title('Mean Absolute Error Over Epochs', fontweight='bold', fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved")

def plot_predictions(y_test, y_pred, test_ids, test_timestamps):
    """Plot predicted vs actual RUL"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual RUL (hours)', fontweight='bold')
    axes[0, 0].set_ylabel('Predicted RUL (hours)', fontweight='bold')
    axes[0, 0].set_title('Predicted vs Actual RUL', fontweight='bold', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Truck 7 time series
    truck7_mask = test_ids == 7
    if np.any(truck7_mask):
        truck7_time = pd.to_datetime(test_timestamps[truck7_mask])
        truck7_actual = y_test[truck7_mask]
        truck7_pred = y_pred[truck7_mask]
        
        axes[0, 1].plot(truck7_time, truck7_actual, 
                        label='Actual RUL', linewidth=2, color='blue')
        axes[0, 1].plot(truck7_time, truck7_pred, 
                        label='Predicted RUL', linewidth=2, color='red', alpha=0.7)
        axes[0, 1].axhline(y=24, color='orange', linestyle='--', 
                          label='24h Warning')
        axes[0, 1].set_xlabel('Timestamp', fontweight='bold')
        axes[0, 1].set_ylabel('RUL (hours)', fontweight='bold')
        axes[0, 1].set_title('Truck 7: RUL Prediction Over Time', 
                             fontweight='bold', fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Residuals
    residuals = y_test - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted RUL (hours)', fontweight='bold')
    axes[1, 0].set_ylabel('Residuals (hours)', fontweight='bold')
    axes[1, 0].set_title('Residual Plot', fontweight='bold', fontsize=13)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Error distribution
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error (hours)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Error Distribution', fontweight='bold', fontsize=13)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('rul_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction plots saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LSTM ANOMALY DETECTION & RUL PREDICTION")
    print("Mansourah-Massarah Gold Mine Haul Truck Fleet")
    print("="*70)
    
    # Train model
    results = train_lstm_model()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_history(results['history'])
    plot_predictions(
        results['y_test'], 
        results['y_pred'], 
        results['test_ids'],
        results['test_timestamps']
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModel saved as: best_lstm_model.h5")
    print("Visualizations saved: training_history.png, rul_predictions.png")
    print("\nNext steps:")
    print("1. Review prediction accuracy on Truck 7")
    print("2. Analyze early warning detection capability")
    print("3. Optimize hyperparameters if needed")
    print("4. Deploy model for real-time monitoring")
