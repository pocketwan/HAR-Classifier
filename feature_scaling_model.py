import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Parameters
WINDOW_SIZE = 200
STEP_SIZE = 25
DATA_PATH = os.path.join('data', 'A_DeviceMotion_data')  # Assumes script in project root
N_SPLITS = 5
EPOCHS = 20
BATCH_SIZE = 64

# Human-readable label mapping
label_map = {
    'dws': 'downstairs',
    'jog': 'jogging',
    'ups': 'upstairs',
    'wlk': 'walking',
    'std': 'standing',
    'sit': 'sitting'
}

def load_data(data_path):
    all_data = []
    for activity in os.listdir(data_path):
        activity_path = os.path.join(data_path, activity)
        if os.path.isdir(activity_path):
            for file in os.listdir(activity_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(activity_path, file))
                    activity_prefix = activity.split('_')[0]
                    df['activity'] = label_map.get(activity_prefix, activity_prefix)
                    all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def normalize_features(df, exclude_cols=['activity']):
    feature_cols = df.columns.difference(exclude_cols)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def segment_data(df, window_size, step_size):
    segments = []
    labels = []
    for i in range(0, len(df) - window_size, step_size):
        segment = df.iloc[i:i+window_size]
        data = segment.drop('activity', axis=1).values
        label = Counter(segment['activity']).most_common(1)[0][0]
        segments.append(data)
        labels.append(label)
    return np.array(segments), np.array(labels)

def three_way_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_fraction = val_size / (1 - test_size)  # adjust for reduced dataset size
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_k_fold_splits(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y)

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion(cm, classes, fold):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold{fold}.png')
    plt.close()

def plot_history(history, fold):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'accuracy_fold{fold}.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_fold{fold}.png')
    plt.close()

# Main script
if __name__ == '__main__':
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    df.ffill(inplace=True)
    df = normalize_features(df, exclude_cols=['activity'])

    print("Segmenting time series...")
    segments, labels = segment_data(df, WINDOW_SIZE, STEP_SIZE)

    print("Encoding labels...")
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    class_names = le.classes_

    print("Computing class weights...")
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(encoded_labels),
                                         y=encoded_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # 3-Way Holdout
    """
    print("Using 3-way holdout split...")
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(segments, categorical_labels)
    model = create_model(input_shape=(WINDOW_SIZE, X_train.shape[2]), num_classes=categorical_labels.shape[1])
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val), class_weight=class_weight_dict, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {acc*100:.2f}%")
    """

    print("Starting cross-validation...")
    skf = get_k_fold_splits(segments, encoded_labels, n_splits=N_SPLITS)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        X_train, X_test = segments[train_idx], segments[test_idx]
        y_train, y_test = categorical_labels[train_idx], categorical_labels[test_idx]

        model = create_model(input_shape=(WINDOW_SIZE, X_train.shape[2]), num_classes=categorical_labels.shape[1])

        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            class_weight=class_weight_dict,
                            verbose=1)

        plot_history(history, fold+1)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy: {acc*100:.2f}%")
        accuracies.append(acc)

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plot_confusion(cm, class_names, fold+1)

        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

	

    print("\nFinal Results:")
    print("Average Accuracy: {:.2f}%".format(np.mean(accuracies) * 100))
    print("Std Deviation: {:.2f}%".format(np.std(accuracies) * 100))
   
    # Accuracy box plot
    plt.figure(figsize=(6, 4))
    plt.boxplot(accuracies, vert=True, patch_artist=True, labels=['Accuracy'])
    plt.title('Cross-Validation Accuracy Distribution')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_boxplot.png')
    plt.show()
