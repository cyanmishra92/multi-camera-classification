"""Enhanced federated learning trainer with actual model training."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from copy import deepcopy

from .federated_trainer import FederatedTrainer
from .simple_cnn_model import SimpleCNNModel, ModelParameters, LocalDataGenerator
from ..core.camera import Camera

logger = logging.getLogger(__name__)


class EnhancedFederatedTrainer(FederatedTrainer):
    """
    Enhanced federated trainer with actual CNN model training.
    
    Features:
    - Real model training with simple CNN
    - Energy-aware participant selection
    - Weighted aggregation based on data quality
    - Adaptive learning rates
    """
    
    def __init__(
        self,
        model_params: Optional[ModelParameters] = None,
        aggregation_method: str = 'weighted_avg',
        min_participants: int = 3,
        participation_threshold: float = 0.3,
        rounds_per_epoch: int = 10,
        local_epochs: int = 5,
        local_batch_size: int = 16
    ):
        """
        Initialize enhanced federated trainer.
        
        Args:
            model_params: Parameters for CNN model
            aggregation_method: Method for aggregating models
            min_participants: Minimum participants per round
            participation_threshold: Energy threshold for participation
            rounds_per_epoch: Training rounds per epoch
            local_epochs: Local training epochs per round
            local_batch_size: Batch size for local training
        """
        super().__init__()
        
        self.model_params = model_params or ModelParameters()
        self.aggregation_method = aggregation_method
        self.min_participants = min_participants
        self.participation_threshold = participation_threshold
        self.rounds_per_epoch = rounds_per_epoch
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        
        # Global model
        self.global_model = SimpleCNNModel(self.model_params)
        
        # Local models for each camera
        self.local_models = {}
        self.data_generators = {}
        
        # Training history
        self.training_history = []
        self.round_count = 0
        
        # Performance tracking
        self.model_accuracy = 0.5  # Initial accuracy
        self.convergence_rate = 0.0
        
    def initialize_camera_models(self, cameras: List[Camera]):
        """Initialize local models for each camera."""
        for camera in cameras:
            # Create local model
            self.local_models[camera.camera_id] = SimpleCNNModel(self.model_params)
            
            # Initialize with global weights
            self.local_models[camera.camera_id].set_model_weights(
                self.global_model.get_model_weights()
            )
            
            # Create data generator
            self.data_generators[camera.camera_id] = LocalDataGenerator(
                camera.camera_id
            )
            
        logger.info(f"Initialized {len(cameras)} local models")
    
    def select_participants(self, cameras: List[Camera]) -> List[Camera]:
        """
        Select cameras to participate in current round.
        
        Args:
            cameras: List of all cameras
            
        Returns:
            List of participating cameras
        """
        # Filter cameras with sufficient energy
        eligible = [
            cam for cam in cameras 
            if cam.current_energy / cam.energy_model.capacity >= self.participation_threshold
        ]
        
        if len(eligible) < self.min_participants:
            # If not enough eligible, take top by energy
            sorted_cameras = sorted(
                cameras, 
                key=lambda c: c.current_energy, 
                reverse=True
            )
            return sorted_cameras[:self.min_participants]
        
        # Randomly sample from eligible cameras
        n_participants = min(len(eligible), max(self.min_participants, int(len(eligible) * 0.4)))
        indices = np.random.choice(len(eligible), n_participants, replace=False)
        
        return [eligible[i] for i in indices]
    
    def local_training(self, camera: Camera, 
                      classification_history: Optional[List[Dict]] = None) -> Dict:
        """
        Perform local training on camera.
        
        Args:
            camera: Camera to train
            classification_history: Recent classification results
            
        Returns:
            Training results including model updates
        """
        camera_id = camera.camera_id
        local_model = self.local_models[camera_id]
        data_gen = self.data_generators[camera_id]
        
        # Start with global model weights
        local_model.set_model_weights(self.global_model.get_model_weights())
        
        # Generate training data
        total_loss = 0.0
        n_batches = self.local_epochs
        
        for epoch in range(self.local_epochs):
            # Generate batch based on camera's view
            x_batch, y_batch = data_gen.generate_batch(self.local_batch_size)
            
            # Train local model
            loss = local_model.train_step(x_batch, y_batch)
            total_loss += loss
        
        # Evaluate local model
        x_test, y_test = data_gen.generate_batch(32)  # Test batch
        eval_results = local_model.evaluate(x_test, y_test)
        
        # Simulate energy consumption for training
        training_energy = self.local_epochs * self.local_batch_size * 0.1
        camera.state.energy -= min(training_energy, camera.current_energy * 0.1)
        
        return {
            'camera_id': camera_id,
            'model_weights': local_model.get_model_weights(),
            'training_loss': total_loss / n_batches,
            'local_accuracy': eval_results['accuracy'],
            'data_samples': self.local_epochs * self.local_batch_size,
            'energy_used': training_energy
        }
    
    def aggregate_models(self, local_updates: List[Dict]) -> Dict:
        """
        Aggregate local model updates.
        
        Args:
            local_updates: List of local training results
            
        Returns:
            Aggregated model weights
        """
        if not local_updates:
            return self.global_model.get_model_weights()
        
        if self.aggregation_method == 'simple_avg':
            return self._simple_average(local_updates)
        elif self.aggregation_method == 'weighted_avg':
            return self._weighted_average(local_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _simple_average(self, local_updates: List[Dict]) -> Dict:
        """Simple averaging of model weights."""
        # Initialize with zeros
        avg_weights = deepcopy(local_updates[0]['model_weights'])
        
        # Zero out weights
        for layer in avg_weights['weights']:
            avg_weights['weights'][layer] *= 0
        for layer in avg_weights['biases']:
            avg_weights['biases'][layer] *= 0
        
        # Sum all weights
        for update in local_updates:
            for layer in update['model_weights']['weights']:
                avg_weights['weights'][layer] += update['model_weights']['weights'][layer]
            for layer in update['model_weights']['biases']:
                avg_weights['biases'][layer] += update['model_weights']['biases'][layer]
        
        # Average
        n = len(local_updates)
        for layer in avg_weights['weights']:
            avg_weights['weights'][layer] /= n
        for layer in avg_weights['biases']:
            avg_weights['biases'][layer] /= n
            
        return avg_weights
    
    def _weighted_average(self, local_updates: List[Dict]) -> Dict:
        """Weighted averaging based on local accuracy and data samples."""
        # Calculate weights
        weights = []
        for update in local_updates:
            # Weight by accuracy and data samples
            w = update['local_accuracy'] * np.sqrt(update['data_samples'])
            weights.append(w)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Initialize with zeros
        avg_weights = deepcopy(local_updates[0]['model_weights'])
        
        # Zero out weights
        for layer in avg_weights['weights']:
            avg_weights['weights'][layer] *= 0
        for layer in avg_weights['biases']:
            avg_weights['biases'][layer] *= 0
        
        # Weighted sum
        for update, w in zip(local_updates, weights):
            for layer in update['model_weights']['weights']:
                avg_weights['weights'][layer] += w * update['model_weights']['weights'][layer]
            for layer in update['model_weights']['biases']:
                avg_weights['biases'][layer] += w * update['model_weights']['biases'][layer]
                
        return avg_weights
    
    def train_round(self, cameras: List[Camera]) -> Dict:
        """
        Execute one round of federated training.
        
        Args:
            cameras: List of all cameras
            
        Returns:
            Training round results
        """
        self.round_count += 1
        logger.info(f"Starting federated training round {self.round_count}")
        
        # Select participants
        participants = self.select_participants(cameras)
        logger.info(f"Selected {len(participants)} participants")
        
        # Local training
        local_updates = []
        for camera in participants:
            try:
                update = self.local_training(camera)
                local_updates.append(update)
            except Exception as e:
                logger.error(f"Local training failed for camera {camera.camera_id}: {e}")
        
        if not local_updates:
            logger.warning("No successful local updates")
            return {
                'round': self.round_count,
                'participants': 0,
                'success': False
            }
        
        # Aggregate models
        aggregated_weights = self.aggregate_models(local_updates)
        
        # Update global model
        self.global_model.set_model_weights(aggregated_weights)
        
        # Evaluate global model
        test_data_gen = LocalDataGenerator(999)  # Test data generator
        x_test, y_test = test_data_gen.generate_batch(100)
        eval_results = self.global_model.evaluate(x_test, y_test)
        
        # Update model accuracy
        self.model_accuracy = eval_results['accuracy']
        
        # Calculate convergence rate
        if len(self.training_history) > 0:
            prev_acc = self.training_history[-1]['global_accuracy']
            self.convergence_rate = abs(self.model_accuracy - prev_acc)
        
        # Record results
        round_results = {
            'round': self.round_count,
            'participants': len(participants),
            'local_accuracies': [u['local_accuracy'] for u in local_updates],
            'global_accuracy': self.model_accuracy,
            'global_loss': eval_results['loss'],
            'convergence_rate': self.convergence_rate,
            'avg_energy_used': np.mean([u['energy_used'] for u in local_updates]),
            'success': True
        }
        
        self.training_history.append(round_results)
        
        return round_results
    
    def get_model_accuracy_boost(self, camera: Camera) -> float:
        """
        Get accuracy boost from federated learning for a camera.
        
        Args:
            camera: Camera to check
            
        Returns:
            Accuracy boost factor (1.0 = no boost)
        """
        if camera.camera_id not in self.local_models:
            return 1.0
        
        # Base boost from global model performance
        base_boost = 1.0 + (self.model_accuracy - 0.5) * 0.3
        
        # Additional boost if camera participated recently
        recent_participants = []
        for hist in self.training_history[-5:]:
            if 'participant_ids' in hist:
                recent_participants.extend(hist['participant_ids'])
        
        participation_bonus = 0.05 if camera.camera_id in recent_participants else 0
        
        return min(1.5, base_boost + participation_bonus)
    
    def should_trigger_training(self, network_accuracy: float, 
                               energy_available: float) -> bool:
        """
        Determine if federated training should be triggered.
        
        Args:
            network_accuracy: Current network accuracy
            energy_available: Average available energy
            
        Returns:
            Whether to trigger training
        """
        # Don't train too frequently
        if self.round_count > 0 and len(self.training_history) > 0:
            last_round = self.training_history[-1]['round']
            if self.round_count - last_round < 10:
                return False
        
        # Trigger if accuracy is low
        if network_accuracy < 0.75:
            return True
        
        # Trigger if convergence has stalled
        if self.convergence_rate < 0.001 and self.model_accuracy < 0.85:
            return True
        
        # Trigger periodically if energy is available
        if energy_available > 0.7 and self.round_count % 20 == 0:
            return True
            
        return False