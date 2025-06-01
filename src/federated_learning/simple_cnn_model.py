"""Simple CNN model for federated learning in camera network."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Parameters for simple CNN model."""
    input_size: int = 64  # Small patches
    hidden_size: int = 32
    output_size: int = 2  # Binary classification
    learning_rate: float = 0.01
    

class SimpleCNNModel:
    """
    Lightweight CNN model suitable for edge devices.
    
    This is a simplified model for demonstration - in practice would use
    actual deep learning framework like TensorFlow Lite or PyTorch Mobile.
    """
    
    def __init__(self, params: ModelParameters = None):
        """Initialize simple CNN model."""
        self.params = params or ModelParameters()
        
        # Initialize weights (simplified)
        self.weights = {
            'conv1': np.random.randn(3, 3, 1, 8) * 0.1,  # 3x3 conv, 1->8 channels
            'conv2': np.random.randn(3, 3, 8, 16) * 0.1,  # 3x3 conv, 8->16 channels
            'fc1': np.random.randn(16 * 16 * 16, self.params.hidden_size) * 0.1,
            'fc2': np.random.randn(self.params.hidden_size, self.params.output_size) * 0.1
        }
        
        self.biases = {
            'conv1': np.zeros(8),
            'conv2': np.zeros(16),
            'fc1': np.zeros(self.params.hidden_size),
            'fc2': np.zeros(self.params.output_size)
        }
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input image patch (64x64x1)
            
        Returns:
            Output logits (2,)
        """
        # Simplified forward pass (no actual convolution for speed)
        # In practice, would use proper convolution operations
        
        # Flatten and pass through fully connected layers
        x_flat = x.flatten()
        
        # Simplified feature extraction
        features = np.tanh(x_flat[:self.params.hidden_size])
        
        # Output layer
        logits = np.dot(features, self.weights['fc2'][:self.params.hidden_size, :]) + self.biases['fc2']
        
        return logits
    
    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction with confidence.
        
        Args:
            x: Input image patch
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        logits = self.forward(x)
        probs = self.softmax(logits)
        
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        return pred_class, confidence
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Single training step (simplified).
        
        Args:
            x_batch: Batch of input patches
            y_batch: Batch of labels
            
        Returns:
            Loss value
        """
        batch_size = len(x_batch)
        total_loss = 0.0
        
        for x, y in zip(x_batch, y_batch):
            # Forward pass
            logits = self.forward(x)
            probs = self.softmax(logits)
            
            # Cross-entropy loss
            loss = -np.log(probs[y] + 1e-7)
            total_loss += loss
            
            # Simplified gradient update (would use backprop in practice)
            # Update output layer only for efficiency
            error = probs.copy()
            error[y] -= 1
            
            # Get features
            x_flat = x.flatten()
            features = np.tanh(x_flat[:self.params.hidden_size])
            
            # Update weights
            grad_w = np.outer(features, error)
            self.weights['fc2'][:self.params.hidden_size, :] -= self.params.learning_rate * grad_w
            self.biases['fc2'] -= self.params.learning_rate * error
        
        avg_loss = total_loss / batch_size
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights for federated aggregation."""
        return {
            'weights': {k: v.copy() for k, v in self.weights.items()},
            'biases': {k: v.copy() for k, v in self.biases.items()}
        }
    
    def set_model_weights(self, weights_dict: Dict[str, Dict[str, np.ndarray]]):
        """Set model weights from federated aggregation."""
        self.weights = {k: v.copy() for k, v in weights_dict['weights'].items()}
        self.biases = {k: v.copy() for k, v in weights_dict['biases'].items()}
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            x_test: Test inputs
            y_test: Test labels
            
        Returns:
            Dictionary with accuracy and loss
        """
        correct = 0
        total_loss = 0.0
        
        for x, y in zip(x_test, y_test):
            pred, conf = self.predict(x)
            if pred == y:
                correct += 1
                
            logits = self.forward(x)
            probs = self.softmax(logits)
            loss = -np.log(probs[y] + 1e-7)
            total_loss += loss
        
        accuracy = correct / len(x_test) if len(x_test) > 0 else 0
        avg_loss = total_loss / len(x_test) if len(x_test) > 0 else 0
        
        self.accuracy_history.append(accuracy)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'confidence': np.mean([self.predict(x)[1] for x in x_test])
        }


class LocalDataGenerator:
    """Generate synthetic training data for cameras."""
    
    def __init__(self, camera_id: int, data_distribution: str = 'uniform'):
        """
        Initialize data generator.
        
        Args:
            camera_id: Camera ID for deterministic generation
            data_distribution: Type of data distribution
        """
        self.camera_id = camera_id
        self.rng = np.random.RandomState(camera_id)
        self.data_distribution = data_distribution
        
    def generate_batch(self, batch_size: int, 
                      object_position: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate batch of synthetic training data.
        
        Args:
            batch_size: Number of samples
            object_position: Optional position for context
            
        Returns:
            Tuple of (inputs, labels)
        """
        # Generate synthetic image patches
        x_batch = []
        y_batch = []
        
        for _ in range(batch_size):
            # Simple synthetic data based on camera view
            if self.rng.random() < 0.5:
                # Class 0: Low frequency pattern
                patch = self.rng.randn(64, 64, 1) * 0.1
                patch[20:40, 20:40] = self.rng.randn(20, 20, 1) * 0.5
                label = 0
            else:
                # Class 1: High frequency pattern
                patch = self.rng.randn(64, 64, 1) * 0.1
                patch[::2, ::2] = self.rng.randn(32, 32, 1) * 0.5
                label = 1
            
            x_batch.append(patch)
            y_batch.append(label)
        
        return np.array(x_batch), np.array(y_batch)