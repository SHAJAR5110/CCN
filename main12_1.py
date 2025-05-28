import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import os
import cv2
from pathlib import Path

class PlantDiseaseDetector:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = []
        self.history = None
        
    def create_model(self, num_classes):
        """Create CNN model for plant disease detection"""
        model = keras.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Rescaling
            layers.Rescaling(1./255),
            
            # CNN layers
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess plant disease dataset"""
        # Create dataset from directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32
        )
        
        self.class_names = train_ds.class_names
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def train_model(self, train_ds, val_ds, epochs=50):
        """Train the CNN model"""
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        return img_array
    
    def predict_disease(self, image_path, show_image=True):
        """Predict plant disease from image"""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return None
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        predicted_disease = self.class_names[predicted_class]
        
        # Display results
        if show_image:
            plt.figure(figsize=(8, 6))
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Predicted: {predicted_disease}\nConfidence: {confidence:.2%}')
            plt.show()
        
        return {
            'disease': predicted_disease,
            'confidence': confidence,
            'all_probabilities': dict(zip(self.class_names, predictions[0]))
        }
    
    def evaluate_model(self, val_ds):
        """Evaluate model performance"""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return
        
        # Get predictions for validation set
        y_true = []
        y_pred = []
        
        for images, labels in val_ds:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        
        # Save class names
        with open(f"{filepath}_classes.txt", "w") as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load class names
        try:
            with open(f"{filepath}_classes.txt", "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print("Class names file not found. Please set class_names manually.")
        
        print(f"Model loaded from {filepath}")

# Example usage and demonstration
def demo_plant_disease_detection():
    """Demonstration of the plant disease detection system"""
    
    # Initialize detector
    detector = PlantDiseaseDetector()
    
    # Example disease classes (common plant diseases)
    example_classes = [
        'Healthy',
        'Apple_Scab',
        'Apple_Black_rot',
        'Apple_Cedar_apple_rust',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Potato_Early_blight',
        'Potato_Late_blight',
        'Corn_Blight',
        'Corn_Common_rust'
    ]
    
    print("Plant Disease Detection System")
    print("=" * 40)
    print(f"Supported diseases: {len(example_classes)} classes")
    print("\nExample classes:")
    for i, class_name in enumerate(example_classes, 1):
        print(f"{i:2d}. {class_name.replace('_', ' ')}")
    
    print("\nModel Architecture Summary:")
    print("- Input: 224x224 RGB images")
    print("- Data augmentation: Random flip, rotation, zoom")
    print("- CNN layers: 5 convolutional blocks with batch normalization")
    print("- Global average pooling")
    print("- Dense layers with dropout for regularization")
    print("- Output: Softmax activation for multi-class classification")
    
    return detector

# Additional utility functions
def create_sample_data_structure():
    """Create sample directory structure for plant disease dataset"""
    structure = """
    Plant Disease Dataset Structure:
    
    data/
    ├── Healthy/
    │   ├── healthy_001.jpg
    │   ├── healthy_002.jpg
    │   └── ...
    ├── Apple_Scab/
    │   ├── scab_001.jpg
    │   ├── scab_002.jpg
    │   └── ...
    ├── Tomato_Early_blight/
    │   ├── blight_001.jpg
    │   ├── blight_002.jpg
    │   └── ...
    └── [other disease classes]/
        └── ...
    
    Each class should have at least 100-500 images for good training results.
    """
    print(structure)

def training_tips():
    """Provide training tips for plant disease detection"""
    tips = """
    Training Tips for Plant Disease Detection:
    
    1. Data Quality:
       - Use high-resolution images (at least 224x224)
       - Ensure good lighting and clear focus
       - Include various angles and stages of disease
    
    2. Data Augmentation:
       - Helps model generalize to new conditions
       - Includes rotation, flipping, zooming
       - Simulates real-world variations
    
    3. Class Balance:
       - Ensure roughly equal samples per class
       - Use techniques like SMOTE for imbalanced data
    
    4. Validation:
       - Keep 20% data for validation
       - Use stratified sampling to maintain class distribution
    
    5. Early Stopping:
       - Prevents overfitting
       - Monitors validation loss
    
    6. Transfer Learning:
       - Consider using pre-trained models (ResNet, EfficientNet)
       - Fine-tune for plant disease specific features
    """
    print(tips)

# Run demonstration
if __name__ == "__main__":
    # Create detector instance
    detector = demo_plant_disease_detection()
    
    # Show sample data structure
    print("\n" + "=" * 50)
    create_sample_data_structure()
    
    # Show training tips
    print("\n" + "=" * 50)
    training_tips()
    
    print("\n" + "=" * 50)
    print("To use this system:")
    print("1. Prepare your dataset in the required structure")
    print("2. Load and train: detector.load_and_preprocess_data('path/to/data')")
    print("3. Train model: detector.train_model(train_ds, val_ds)")
    print("4. Make predictions: detector.predict_disease('path/to/image.jpg')")
    print("5. Save model: detector.save_model('plant_disease_model')")