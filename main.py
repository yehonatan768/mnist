import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time
import pickle
import json
import os
from datetime import datetime
import pandas as pd

# Import your modules
from preprocessing import (
    normalize_images, binarize_images, apply_pca,
    extract_hog_features, zca_whitening, augment_images,
    edge_detection
)
from models import (
    train_knn, train_svm, train_random_forest,
    build_mlp, build_cnn, evaluate_model
)


# Create directories for saving results
def create_directories():
    dirs = ['models', 'figures', 'results', 'preprocessed_data']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def load_and_prepare_data():
    """Load MNIST dataset and prepare train/test splits"""
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def prepare_data_for_models(X_train, X_test, preprocessing_method=None):
    """Prepare data based on preprocessing method"""
    if preprocessing_method == 'sobel':
        print("Applying Sobel edge detection...")
        X_train_processed = edge_detection(X_train, method='sobel')
        X_test_processed = edge_detection(X_test, method='sobel')
    elif preprocessing_method == 'hog':
        print("Extracting HOG features...")
        X_train_processed = extract_hog_features(X_train)
        X_test_processed = extract_hog_features(X_test)
    elif preprocessing_method == 'pca':
        print("Applying PCA...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train_processed, pca = apply_pca(X_train, n_components=100)
        X_test_processed = pca.transform(X_test_flat)
    elif preprocessing_method == 'binarize':
        print("Binarizing images...")
        X_train_processed = binarize_images(X_train, threshold=0.5)
        X_test_processed = binarize_images(X_test, threshold=0.5)
    else:
        # No preprocessing, just flatten for non-CNN models
        X_train_processed = X_train
        X_test_processed = X_test

    return X_train_processed, X_test_processed


def flatten_for_classical(X_train, X_test):
    """Flatten images for classical ML models"""
    return X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)


def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessing_name="baseline"):
    """Train all models and collect results"""
    results = {}
    models = {}

    # Prepare data for classical models (flatten if needed)
    if len(X_train.shape) > 2:
        X_train_flat, X_test_flat = flatten_for_classical(X_train, X_test)
    else:
        X_train_flat, X_test_flat = X_train, X_test

    # 1. K-Nearest Neighbors
    print("\n" + "=" * 50)
    print("Training K-Nearest Neighbors...")
    start_time = time.time()
    knn_model = train_knn(X_train_flat, y_train, n_neighbors=5)
    knn_train_time = time.time() - start_time

    knn_pred = knn_model.predict(X_test_flat)
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"KNN Accuracy: {knn_acc:.4f}")
    print(f"Training time: {knn_train_time:.2f} seconds")

    results['KNN'] = {
        'accuracy': knn_acc,
        'train_time': knn_train_time,
        'predictions': knn_pred,
        'preprocessing': preprocessing_name
    }
    models['KNN'] = knn_model

    # 2. Support Vector Machine
    print("\n" + "=" * 50)
    print("Training Support Vector Machine...")
    print("Note: This may take several minutes...")
    start_time = time.time()

    # Use a subset for SVM if dataset is too large
    subset_size = min(10000, len(X_train_flat))
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    X_train_svm = X_train_flat[indices]
    y_train_svm = y_train[indices]

    svm_model = train_svm(X_train_svm, y_train_svm, kernel='rbf')
    svm_train_time = time.time() - start_time

    svm_pred = svm_model.predict(X_test_flat)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"Training time: {svm_train_time:.2f} seconds")

    results['SVM'] = {
        'accuracy': svm_acc,
        'train_time': svm_train_time,
        'predictions': svm_pred,
        'preprocessing': preprocessing_name,
        'subset_size': subset_size
    }
    models['SVM'] = svm_model

    # 3. Random Forest
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    start_time = time.time()
    rf_model = train_random_forest(X_train_flat, y_train, n_estimators=100)
    rf_train_time = time.time() - start_time

    rf_pred = rf_model.predict(X_test_flat)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Training time: {rf_train_time:.2f} seconds")

    results['Random Forest'] = {
        'accuracy': rf_acc,
        'train_time': rf_train_time,
        'predictions': rf_pred,
        'preprocessing': preprocessing_name
    }
    models['Random Forest'] = rf_model

    # 4. Multi-Layer Perceptron
    print("\n" + "=" * 50)
    print("Training Multi-Layer Perceptron...")

    # Prepare data for MLP
    if len(X_train.shape) == 2:
        input_shape = (X_train.shape[1],)
    else:
        input_shape = X_train.shape[1:]

    mlp_model = build_mlp(input_shape, num_classes=10)

    start_time = time.time()
    history_mlp = mlp_model.fit(
        X_train, to_categorical(y_train),
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    mlp_train_time = time.time() - start_time

    mlp_pred = np.argmax(mlp_model.predict(X_test), axis=1)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    print(f"MLP Accuracy: {mlp_acc:.4f}")
    print(f"Training time: {mlp_train_time:.2f} seconds")

    results['MLP'] = {
        'accuracy': mlp_acc,
        'train_time': mlp_train_time,
        'predictions': mlp_pred,
        'preprocessing': preprocessing_name,
        'history': history_mlp.history
    }
    models['MLP'] = mlp_model

    # 5. Convolutional Neural Network (only for image data)
    if len(X_train.shape) == 3 or (len(X_train.shape) == 2 and preprocessing_name == "baseline"):
        print("\n" + "=" * 50)
        print("Training Convolutional Neural Network...")

        # Reshape data for CNN if needed
        if len(X_train.shape) == 2:
            size = int(np.sqrt(X_train.shape[1]))
            X_train_cnn = X_train.reshape(-1, size, size, 1)
            X_test_cnn = X_test.reshape(-1, size, size, 1)
        else:
            X_train_cnn = X_train[..., np.newaxis]
            X_test_cnn = X_test[..., np.newaxis]

        cnn_model = build_cnn((28, 28, 1), num_classes=10)

        start_time = time.time()
        history_cnn = cnn_model.fit(
            X_train_cnn, to_categorical(y_train),
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )
        cnn_train_time = time.time() - start_time

        cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
        cnn_acc = accuracy_score(y_test, cnn_pred)
        print(f"CNN Accuracy: {cnn_acc:.4f}")
        print(f"Training time: {cnn_train_time:.2f} seconds")

        results['CNN'] = {
            'accuracy': cnn_acc,
            'train_time': cnn_train_time,
            'predictions': cnn_pred,
            'preprocessing': preprocessing_name,
            'history': history_cnn.history
        }
        models['CNN'] = cnn_model

    return results, models


def analyze_misclassifications(results, y_test):
    """Analyze which digits are most commonly misclassified"""
    misclass_analysis = {}

    for model_name, result in results.items():
        predictions = result['predictions']
        misclassified = y_test != predictions

        # Count misclassifications by true label
        misclass_by_digit = {}
        confusion_pairs = {}

        for digit in range(10):
            digit_mask = y_test == digit
            digit_misclass = misclassified[digit_mask]
            misclass_rate = np.mean(digit_misclass)
            misclass_by_digit[digit] = misclass_rate

            # Find what these digits were misclassified as
            if np.any(digit_misclass):
                wrong_preds = predictions[digit_mask][digit_misclass]
                unique, counts = np.unique(wrong_preds, return_counts=True)
                confusion_pairs[digit] = dict(zip(unique, counts))

        misclass_analysis[model_name] = {
            'by_digit': misclass_by_digit,
            'confusion_pairs': confusion_pairs
        }

    return misclass_analysis


def create_visualizations(results, misclass_analysis, y_test):
    """Create all visualizations"""

    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 8))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Training time comparison
    plt.subplot(2, 2, 2)
    train_times = [results[m]['train_time'] for m in models]
    bars = plt.bar(models, train_times, color=colors)
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45)
    plt.yscale('log')

    # Confusion matrix for best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    cm = confusion_matrix(y_test, results[best_model]['predictions'])

    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model}')

    # Misclassification rates by digit
    plt.subplot(2, 2, 4)
    for model in models[:3]:  # Show top 3 models
        misclass_rates = [misclass_analysis[model]['by_digit'][i] for i in range(10)]
        plt.plot(range(10), misclass_rates, marker='o', label=model)

    plt.xlabel('Digit')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rates by Digit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Training history for neural networks
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in ['MLP', 'CNN']:
        if model_name in results and 'history' in results[model_name]:
            history = results[model_name]['history']

            # Accuracy plot
            axes[0].plot(history['accuracy'], label=f'{model_name} Train')
            axes[0].plot(history['val_accuracy'], label=f'{model_name} Val', linestyle='--')

            # Loss plot
            axes[1].plot(history['loss'], label=f'{model_name} Train')
            axes[1].plot(history['val_loss'], label=f'{model_name} Val', linestyle='--')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Neural Network Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Neural Network Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/nn_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Most confused digit pairs
    plt.figure(figsize=(15, 5))

    for idx, model in enumerate(models[:3]):
        plt.subplot(1, 3, idx + 1)

        # Find top confused pairs
        confused_pairs = []
        for digit, confusions in misclass_analysis[model]['confusion_pairs'].items():
            for confused_as, count in confusions.items():
                confused_pairs.append((digit, confused_as, count))

        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = confused_pairs[:10]

        if top_pairs:
            labels = [f'{p[0]}→{p[1]}' for p in top_pairs]
            counts = [p[2] for p in top_pairs]

            plt.bar(labels, counts, color=colors[idx])
            plt.xlabel('Digit Pair (True→Predicted)')
            plt.ylabel('Count')
            plt.title(f'Top Confused Pairs - {model}')
            plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('figures/confused_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, models, misclass_analysis, preprocessing_name):
    """Save all results and models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save numerical results
    results_summary = {
        'timestamp': timestamp,
        'preprocessing': preprocessing_name,
        'model_results': {}
    }

    for model_name, result in results.items():
        results_summary['model_results'][model_name] = {
            'accuracy': float(result['accuracy']),
            'train_time': float(result['train_time']),
            'preprocessing': result['preprocessing']
        }
        if 'subset_size' in result:
            results_summary['model_results'][model_name]['subset_size'] = result['subset_size']

    # Save to JSON
    with open(f'results/results_{preprocessing_name}_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    # Save models
    for model_name, model in models.items():
        if model_name in ['MLP', 'CNN']:
            # Save Keras models
            model.save(f'models/{model_name}_{preprocessing_name}_{timestamp}.h5')
        else:
            # Save sklearn models
            with open(f'models/{model_name}_{preprocessing_name}_{timestamp}.pkl', 'wb') as f:
                pickle.dump(model, f)

    # Save detailed report
    with open(f'results/detailed_report_{preprocessing_name}_{timestamp}.txt', 'w') as f:
        f.write(f"MNIST Analysis Report - {preprocessing_name}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Model Performance Summary:\n")
        f.write("-" * 30 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Training Time: {result['train_time']:.2f} seconds\n")
            if 'subset_size' in result:
                f.write(f"  Training Subset Size: {result['subset_size']}\n")

        f.write("\n\nMisclassification Analysis:\n")
        f.write("-" * 30 + "\n")
        for model_name, analysis in misclass_analysis.items():
            f.write(f"\n{model_name}:\n")
            f.write("  Misclassification rates by digit:\n")
            for digit, rate in analysis['by_digit'].items():
                f.write(f"    Digit {digit}: {rate:.3f}\n")

    print(f"\nResults saved with timestamp: {timestamp}")


def main():
    """Main execution function"""
    # Create necessary directories
    create_directories()

    # Load data
    X_train, y_train, X_test, y_test = load_and_prepare_data()

    X_train, y_train, X_test, y_test = X_train[:200], y_train[:200], X_test[:200], y_test[:200]

    # Dictionary to store all results
    all_results = {}

    # Test different preprocessing methods
    preprocessing_methods = [
        ('baseline', None),
        ('sobel', 'sobel'),
        ('hog', 'hog'),
        ('pca', 'pca'),
        ('binarize', 'binarize')
    ]

    for prep_name, prep_method in preprocessing_methods:
        print(f"\n{'=' * 60}")
        print(f"Testing with preprocessing: {prep_name}")
        print(f"{'=' * 60}")

        # Prepare data
        X_train_proc, X_test_proc = prepare_data_for_models(
            X_train, X_test, prep_method
        )

        # Train and evaluate models
        results, models = train_and_evaluate_models(
            X_train_proc, y_train, X_test_proc, y_test, prep_name
        )

        # Analyze misclassifications
        misclass_analysis = analyze_misclassifications(results, y_test)

        # Create visualizations
        create_visualizations(results, misclass_analysis, y_test)

        # Save results
        save_results(results, models, misclass_analysis, prep_name)

        # Store for comparison
        all_results[prep_name] = results

    # Create final comparison visualization
    create_preprocessing_comparison(all_results)

    print("\n" + "=" * 60)
    print("Analysis complete! Check the 'figures', 'models', and 'results' directories.")


def create_preprocessing_comparison(all_results):
    """Create visualization comparing all preprocessing methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Get all unique models
    all_models = set()
    for results in all_results.values():
        all_models.update(results.keys())
    all_models = sorted(list(all_models))

    # Prepare data for comparison
    prep_methods = list(all_results.keys())

    # 1. Accuracy comparison heatmap
    ax = axes[0, 0]
    accuracy_matrix = []
    for model in all_models:
        row = []
        for prep in prep_methods:
            if model in all_results[prep]:
                row.append(all_results[prep][model]['accuracy'])
            else:
                row.append(np.nan)
        accuracy_matrix.append(row)

    accuracy_matrix = np.array(accuracy_matrix)
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f',
                xticklabels=prep_methods, yticklabels=all_models,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Model Accuracy by Preprocessing Method')
    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Model')

    # 2. Best model for each preprocessing
    ax = axes[0, 1]
    best_models = []
    best_accuracies = []
    for prep in prep_methods:
        best_model = max(all_results[prep].keys(),
                         key=lambda x: all_results[prep][x]['accuracy'])
        best_models.append(best_model)
        best_accuracies.append(all_results[prep][best_model]['accuracy'])

    bars = ax.bar(prep_methods, best_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(prep_methods))))
    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('Best Model Performance by Preprocessing')
    ax.set_ylim(0.9, 1.0)

    # Add model names on bars
    for i, (bar, model) in enumerate(zip(bars, best_models)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                model, ha='center', va='bottom', rotation=0, fontsize=9)

    # 3. Training time comparison
    ax = axes[1, 0]
    width = 0.15
    x = np.arange(len(prep_methods))

    for i, model in enumerate(all_models[:5]):  # Show top 5 models
        times = []
        for prep in prep_methods:
            if model in all_results[prep]:
                times.append(all_results[prep][model]['train_time'])
            else:
                times.append(0)

        ax.bar(x + i * width, times, width, label=model)

    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(prep_methods)
    ax.legend()
    ax.set_yscale('log')

    # 4. Improvement over baseline
    ax = axes[1, 1]
    baseline_results = all_results['baseline']

    for model in all_models:
        if model in baseline_results:
            baseline_acc = baseline_results[model]['accuracy']
            improvements = []

            for prep in prep_methods[1:]:  # Skip baseline
                if model in all_results[prep]:
                    improvement = (all_results[prep][model]['accuracy'] - baseline_acc) * 100
                    improvements.append(improvement)
                else:
                    improvements.append(0)

            ax.plot(prep_methods[1:], improvements, marker='o', label=model)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('Performance Improvement by Preprocessing')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a summary table
    summary_df = pd.DataFrame(index=all_models, columns=prep_methods)
    for model in all_models:
        for prep in prep_methods:
            if model in all_results[prep]:
                summary_df.loc[model, prep] = f"{all_results[prep][model]['accuracy']:.3f}"
            else:
                summary_df.loc[model, prep] = "N/A"

    # Save summary table
    summary_df.to_csv('results/accuracy_summary_table.csv')
    print("\nAccuracy Summary Table:")
    print(summary_df)


if __name__ == "__main__":
    main()
