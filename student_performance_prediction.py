import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. DATA GENERATION
# ============================================================
def generate_data(n_samples=1000):
    """Generate synthetic student performance data."""
    print("=" * 60)
    print("STEP 1: Generating Synthetic Dataset")
    print("=" * 60)

    # Features: Study Hours (0 to 10), Attendance (40% to 100%)
    study_hours = np.random.uniform(0, 10, n_samples)
    attendance = np.random.uniform(40, 100, n_samples)

    # Pass/Fail logic:
    # A weighted combination of study hours and attendance with noise
    # Higher study hours and higher attendance -> more likely to pass
    noise = np.random.normal(0, 1, n_samples)
    score = (0.7 * study_hours) + (0.05 * attendance) + noise
    pass_fail = (score > 8.0).astype(int)

    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Pass_Fail': pass_fail
    })

    print(f"Total samples: {len(df)}")
    print(f"\nClass Distribution:")
    print(f"  Fail (0): {(df['Pass_Fail'] == 0).sum()}")
    print(f"  Pass (1): {(df['Pass_Fail'] == 1).sum()}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print()
    return df

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
def preprocess_data(df):
    """Split data and apply StandardScaler normalization."""
    print("=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)

    X = df[['Study_Hours', 'Attendance']].values
    y = df['Pass_Fail'].values

    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size:     {X_test.shape[0]}")

    # Feature Scaling using StandardScaler (Z-score normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nAfter scaling (training set):")
    print(f"  Study_Hours -> mean: {X_train_scaled[:, 0].mean():.4f}, std: {X_train_scaled[:, 0].std():.4f}")
    print(f"  Attendance  -> mean: {X_train_scaled[:, 1].mean():.4f}, std: {X_train_scaled[:, 1].std():.4f}")
    print()
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================
# 3. MODEL BUILDING (Keras Neural Network)
# ============================================================
def build_model():
    """Build and compile a Keras Sequential neural network for binary classification."""
    print("=" * 60)
    print("STEP 3: Building Keras Neural Network")
    print("=" * 60)

    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(2,), name='hidden_layer_1'),
        layers.Dense(8, activation='relu', name='hidden_layer_2'),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()
    print()
    return model

# ============================================================
# 4. VISUALIZATION
# ============================================================
def plot_data_distribution(df):
    """Plot scatter plot of the raw data colored by Pass/Fail."""
    plt.figure(figsize=(8, 6))
    colors = {0: '#e74c3c', 1: '#2ecc71'}
    labels = {0: 'Fail', 1: 'Pass'}
    for cls in [0, 1]:
        subset = df[df['Pass_Fail'] == cls]
        plt.scatter(subset['Study_Hours'], subset['Attendance'],
                    c=colors[cls], label=labels[cls], alpha=0.6, edgecolors='w', s=50)
    plt.xlabel('Study Hours (per week)')
    plt.ylabel('Attendance (%)')
    plt.title('Student Data: Study Hours vs Attendance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150)
    print("Saved: data_distribution.png")
    plt.show()


def plot_learning_curves(history):
    """Plot training & validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    print("Saved: learning_curves.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix as a heatmap using seaborn."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail (0)', 'Pass (1)'],
                yticklabels=['Fail (0)', 'Pass (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Saved: confusion_matrix.png")
    plt.show()


# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def main():
    # --- Step 1: Generate Data ---
    df = generate_data()

    # --- Visualize raw data ---
    plot_data_distribution(df)

    # --- Step 2: Preprocess ---
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # --- Step 3: Build Model ---
    model = build_model()

    # --- Step 4: Train Model ---
    print("=" * 60)
    print("STEP 4: Training the Model")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    print()

    # --- Step 5: Evaluate Model ---
    print("=" * 60)
    print("STEP 5: Model Evaluation")
    print("=" * 60)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

    # Confusion Matrix (printed)
    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(cm)
    print()

    # --- Visualize results ---
    plot_learning_curves(history)
    plot_confusion_matrix(y_test, y_pred)

    # --- Save model ---
    model.save('student_performance_model.keras')
    print("Model saved as student_performance_model.keras")

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
