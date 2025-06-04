"""Game-theoretic enhanced fixed frequency algorithm."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .enhanced_fixed_frequency import EnhancedFixedFrequencyAlgorithm
from ..core.camera import Camera
from ..game_theory.strategic_agent import StrategicAgent
from ..game_theory.nash_equilibrium import NashEquilibriumSolver
from ..game_theory.utility_functions import UtilityParameters

logger = logging.getLogger(__name__)


class GameTheoreticFixedFrequencyAlgorithm(EnhancedFixedFrequencyAlgorithm):
    """
    Fixed frequency algorithm with full game-theoretic camera selection.
    
    Features:
    - Nash equilibrium-based selection within each class
    - Strategic agents with utility maximization
    - Adaptive threshold updates
    - Social welfare optimization
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        utility_params: Optional[UtilityParameters] = None,
        position_weight: float = 0.7,
        use_nash_equilibrium: bool = True,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize game-theoretic fixed frequency algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            utility_params: Parameters for utility functions
            position_weight: Weight for position-based selection
            use_nash_equilibrium: Whether to use Nash equilibrium
            convergence_threshold: Threshold for Nash equilibrium convergence
        """
        # Initialize parent with game theory enabled
        super().__init__(
            cameras=cameras,
            num_classes=num_classes,
            min_accuracy_threshold=min_accuracy_threshold,
            history_length=history_length,
            use_game_theory=True,
            position_weight=position_weight
        )
        
        self.use_nash_equilibrium = use_nash_equilibrium
        self.utility_params = utility_params or UtilityParameters()
        
        # Create strategic agents for each camera
        self.strategic_agents = [
            StrategicAgent(camera, self.utility_params)
            for camera in self.cameras
        ]
        
        # Nash equilibrium solver
        self.nash_solver = NashEquilibriumSolver(
            convergence_threshold=convergence_threshold
        )
        
        # Performance tracking
        self.equilibrium_history = []
        self.welfare_history = []
        
    def _select_strategic_enhanced(self, class_id: int,
                                  object_position: np.ndarray) -> List[int]:
        """
        Strategic selection using Nash equilibrium and position awareness.
        
        Args:
            class_id: Active class ID
            object_position: Object position
            
        Returns:
            Selected camera IDs
        """
        candidate_cameras = self.camera_classes[class_id]

        if not self.use_nash_equilibrium:
            # Fall back to parent implementation
            return super()._select_strategic_enhanced(class_id, object_position)
        
        # Get strategic agents for candidates
        candidate_agents = [self.strategic_agents[i] for i in candidate_cameras]
        
        # Update future value estimates
        for agent in candidate_agents:
            agent.update_future_value()
        
        # Prepare network state with position information
        network_state = self._prepare_network_state(candidate_cameras, object_position)
        
        # Find Nash equilibrium
        equilibrium_actions, converged = self.nash_solver.find_equilibrium(
            candidate_agents, network_state
        )
        
        if not converged:
            logger.warning("Nash equilibrium did not converge, using best effort")
        
        # Get selected cameras from equilibrium
        selected = [
            candidate_cameras[i]
            for i, participates in enumerate(equilibrium_actions)
            if participates
        ]
        
        # Check if we meet accuracy threshold
        if selected:
            selected_cameras = [self.cameras[i] for i in selected]
            collective_accuracy = self._calculate_enhanced_collective_accuracy(
                selected_cameras, object_position
            )
            
            # If below threshold, add more cameras strategically
            if collective_accuracy < self.min_accuracy_threshold:
                selected = self._augment_selection(
                    selected, candidate_cameras, object_position, collective_accuracy
                )
        else:
            # No cameras selected by Nash equilibrium, use fallback
            logger.warning("Nash equilibrium selected no cameras, using fallback")
            selected = self._fallback_selection(candidate_cameras, object_position)
        
        # Analyze equilibrium
        if self.use_nash_equilibrium and selected:
            self._analyze_equilibrium(candidate_agents, equilibrium_actions)
        
        return selected
    
    def _prepare_network_state(self, candidate_cameras: List[int],
                              object_position: np.ndarray) -> Dict:
        """
        Prepare network state information for strategic agents.
        
        Args:
            candidate_cameras: Candidate camera IDs
            object_position: Object position
            
        Returns:
            Network state dictionary
        """
        # Get recent participation history
        recent_participants = self.get_recent_participants()
        recent_in_class = [
            cam_id for cam_id in recent_participants
            if cam_id in candidate_cameras
        ]
        
        # Calculate position-based information
        position_scores = {}
        if self.using_enhanced:
            for cam_id in candidate_cameras:
                camera = self.cameras[cam_id]
                accuracy = camera.accuracy_model.get_position_based_accuracy(
                    camera, object_position
                )
                position_scores[cam_id] = accuracy
        
        # Class performance metrics
        class_id = self.cameras[candidate_cameras[0]].class_assignment
        class_perf = self.get_class_performance(class_id)
        
        return {
            'object_position': object_position,
            'recent_participants': recent_in_class,
            'position_scores': position_scores,
            'class_performance': class_perf,
            'total_classifications': self.total_classifications,
            'recent_accuracy': class_perf.get('accuracy', 0.0)
        }
    
    def _augment_selection(self, current_selection: List[int],
                          candidate_cameras: List[int],
                          object_position: np.ndarray,
                          current_accuracy: float) -> List[int]:
        """
        Augment selection to meet accuracy threshold.
        
        Args:
            current_selection: Currently selected cameras
            candidate_cameras: All candidate cameras
            object_position: Object position
            current_accuracy: Current collective accuracy
            
        Returns:
            Augmented selection
        """
        remaining = [c for c in candidate_cameras if c not in current_selection]
        
        # Score remaining cameras by expected accuracy contribution
        scores = []
        for cam_id in remaining:
            camera = self.cameras[cam_id]
            agent = self.strategic_agents[cam_id]
            
            if camera.can_classify():
                # Position-based accuracy
                if self.using_enhanced:
                    accuracy = camera.accuracy_model.get_position_based_accuracy(
                        camera, object_position
                    )
                else:
                    accuracy = camera.current_accuracy
                
                # Strategic value
                strategic_value = agent.future_value_estimate
                
                # Combined score
                score = 0.7 * accuracy + 0.3 * strategic_value
                scores.append((cam_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add cameras until threshold met
        augmented = current_selection.copy()
        
        for cam_id, _ in scores:
            augmented.append(cam_id)
            
            selected_cameras = [self.cameras[i] for i in augmented]
            new_accuracy = self._calculate_enhanced_collective_accuracy(
                selected_cameras, object_position
            )
            
            if new_accuracy >= self.min_accuracy_threshold:
                break
        
        return augmented
    
    def _fallback_selection(self, candidate_cameras: List[int],
                           object_position: np.ndarray) -> List[int]:
        """
        Fallback selection when Nash equilibrium fails.
        
        Args:
            candidate_cameras: Candidate camera IDs
            object_position: Object position
            
        Returns:
            Selected camera IDs
        """
        # Use position-aware greedy selection
        return self._select_position_aware(candidate_cameras, object_position)
    
    def _analyze_equilibrium(self, agents: List[StrategicAgent],
                           equilibrium_actions: List[bool]) -> None:
        """
        Analyze and record equilibrium properties.
        
        Args:
            agents: Strategic agents involved
            equilibrium_actions: Equilibrium action profile
        """
        analysis = self.nash_solver.analyze_equilibrium(agents, equilibrium_actions)
        
        self.equilibrium_history.append({
            'timestamp': self.total_classifications,
            'participation_rate': analysis['participation_rate'],
            'collective_accuracy': analysis['collective_accuracy'],
            'num_participating': analysis['num_participating']
        })
        
        self.welfare_history.append({
            'timestamp': self.total_classifications,
            'social_welfare': analysis['social_welfare'],
            'optimal_welfare': analysis['optimal_welfare'],
            'price_of_anarchy': analysis['price_of_anarchy']
        })
        
        # Log if price of anarchy is high
        if analysis['price_of_anarchy'] > 1.5:
            logger.warning(f"High price of anarchy: {analysis['price_of_anarchy']:.2f}")
    
    def update_agent_thresholds(self) -> None:
        """Update participation thresholds for all agents."""
        if not self.equilibrium_history:
            return
        
        # Calculate recent participation rate
        recent_rates = [
            eq['participation_rate'] 
            for eq in self.equilibrium_history[-10:]
        ]
        avg_rate = np.mean(recent_rates)
        
        # Target rate based on energy sustainability
        target_rate = 0.3  # Conservative target
        
        # Update all agents
        for agent in self.strategic_agents:
            agent.update_threshold(avg_rate, target_rate)
    
    def get_game_theory_metrics(self) -> Dict:
        """Get game theory specific performance metrics."""
        base_metrics = self.get_performance_metrics()
        
        if not self.equilibrium_history:
            return base_metrics
        
        # Recent equilibrium statistics
        recent_eq = self.equilibrium_history[-20:]
        avg_participation = np.mean([eq['participation_rate'] for eq in recent_eq])
        avg_nash_accuracy = np.mean([eq['collective_accuracy'] for eq in recent_eq])
        
        # Welfare statistics
        if self.welfare_history:
            recent_welfare = self.welfare_history[-20:]
            avg_social_welfare = np.mean([w['social_welfare'] for w in recent_welfare])
            avg_poa = np.mean([w['price_of_anarchy'] for w in recent_welfare 
                              if w['price_of_anarchy'] < float('inf')])
        else:
            avg_social_welfare = 0.0
            avg_poa = 1.0
        
        game_metrics = {
            **base_metrics,
            'avg_participation_rate': avg_participation,
            'avg_nash_accuracy': avg_nash_accuracy,
            'avg_social_welfare': avg_social_welfare,
            'avg_price_of_anarchy': avg_poa,
            'equilibrium_convergence_rate': sum(1 for eq in self.equilibrium_history) / len(self.equilibrium_history)
        }
        
        return game_metrics