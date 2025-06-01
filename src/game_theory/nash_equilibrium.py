"""Nash equilibrium computation and analysis."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from copy import deepcopy

from .strategic_agent import StrategicAgent

logger = logging.getLogger(__name__)


class NashEquilibriumSolver:
    """
    Solver for finding Nash equilibrium in multi-camera game.
    
    Implements iterative best-response dynamics.
    """
    
    def __init__(
        self,
        convergence_threshold: float = 0.01,
        max_iterations: int = 100
    ):
        """
        Initialize Nash equilibrium solver.
        
        Args:
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum iterations for convergence
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
    def find_equilibrium(
        self,
        agents: List[StrategicAgent],
        network_state: Dict
    ) -> Tuple[List[bool], bool]:
        """
        Find Nash equilibrium using best-response dynamics.
        
        Args:
            agents: List of strategic agents
            network_state: Current network state
            
        Returns:
            Tuple of (equilibrium_actions, converged)
        """
        n_agents = len(agents)
        current_actions = [False] * n_agents
        converged = False
        
        for iteration in range(self.max_iterations):
            previous_actions = current_actions.copy()
            
            # Each agent computes best response
            for i, agent in enumerate(agents):
                # Get other agents' actions
                other_actions = current_actions[:i] + current_actions[i+1:]
                
                # Update network state with current actions
                participating_accuracies = [
                    agents[j].camera.current_accuracy 
                    for j in range(n_agents) 
                    if j != i and current_actions[j]
                ]
                
                agent_network_state = {
                    **network_state,
                    'participating_accuracies': participating_accuracies
                }
                
                # Compute best response
                should_participate, _ = agent.decide_participation(
                    current_state={'energy': agent.camera.current_energy},
                    network_state=agent_network_state
                )
                
                current_actions[i] = should_participate
            
            # Check convergence
            if current_actions == previous_actions:
                converged = True
                logger.info(f"Nash equilibrium found in {iteration + 1} iterations")
                break
                
        if not converged:
            logger.warning(f"Nash equilibrium not found in {self.max_iterations} iterations")
            
        return current_actions, converged
    
    def compute_social_welfare(
        self,
        agents: List[StrategicAgent],
        actions: List[bool]
    ) -> float:
        """
        Compute social welfare (sum of utilities) for given actions.
        
        Args:
            agents: List of strategic agents
            actions: List of participation decisions
            
        Returns:
            Total social welfare
        """
        total_welfare = 0.0
        
        participating_accuracies = [
            agents[i].camera.current_accuracy
            for i in range(len(agents))
            if actions[i]
        ]
        
        for i, (agent, participates) in enumerate(zip(agents, actions)):
            if participates:
                # Calculate utility for participating agent
                other_accuracies = [
                    a for j, a in enumerate(participating_accuracies)
                    if j != i
                ]
                
                network_state = {'participating_accuracies': other_accuracies}
                _, utility = agent.decide_participation(
                    current_state={'energy': agent.camera.current_energy},
                    network_state=network_state
                )
                
                total_welfare += utility
            else:
                # Non-participation utility
                utility = agent.utility_function.expected_utility_not_participate(
                    future_value=agent.future_value_estimate
                )
                total_welfare += utility
                
        return total_welfare
    
    def analyze_equilibrium(
        self,
        agents: List[StrategicAgent],
        equilibrium_actions: List[bool]
    ) -> Dict:
        """
        Analyze properties of the equilibrium.
        
        Args:
            agents: List of strategic agents
            equilibrium_actions: Equilibrium action profile
            
        Returns:
            Dictionary with analysis results
        """
        # Compute various metrics
        participation_rate = sum(equilibrium_actions) / len(equilibrium_actions)
        
        # Get participating cameras' energies and accuracies
        participating_energies = [
            agents[i].camera.current_energy
            for i in range(len(agents))
            if equilibrium_actions[i]
        ]
        
        participating_accuracies = [
            agents[i].camera.current_accuracy
            for i in range(len(agents))
            if equilibrium_actions[i]
        ]
        
        # Calculate collective accuracy
        if participating_accuracies:
            collective_accuracy = 1 - np.prod([1 - a for a in participating_accuracies])
        else:
            collective_accuracy = 0.0
            
        # Social welfare
        social_welfare = self.compute_social_welfare(agents, equilibrium_actions)
        
        # Price of anarchy (compare to optimal)
        optimal_actions = self._find_optimal_actions(agents)
        optimal_welfare = self.compute_social_welfare(agents, optimal_actions)
        price_of_anarchy = optimal_welfare / social_welfare if social_welfare > 0 else float('inf')
        
        return {
            'participation_rate': participation_rate,
            'collective_accuracy': collective_accuracy,
            'avg_participating_energy': np.mean(participating_energies) if participating_energies else 0,
            'social_welfare': social_welfare,
            'optimal_welfare': optimal_welfare,
            'price_of_anarchy': price_of_anarchy,
            'num_participating': sum(equilibrium_actions)
        }
    
    def _find_optimal_actions(self, agents: List[StrategicAgent]) -> List[bool]:
        """
        Find socially optimal action profile (brute force for small networks).
        
        Args:
            agents: List of strategic agents
            
        Returns:
            Optimal action profile
        """
        n_agents = len(agents)
        
        if n_agents > 10:
            # Too many agents for brute force, use greedy heuristic
            return self._greedy_optimal(agents)
            
        # Brute force search
        best_welfare = -float('inf')
        best_actions = [False] * n_agents
        
        # Try all 2^n combinations
        for i in range(2**n_agents):
            actions = [(i >> j) & 1 == 1 for j in range(n_agents)]
            
            # Check if all participating agents have enough energy
            valid = all(
                not actions[j] or agents[j].camera.can_classify()
                for j in range(n_agents)
            )
            
            if valid:
                welfare = self.compute_social_welfare(agents, actions)
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_actions = actions
                    
        return best_actions
    
    def _greedy_optimal(self, agents: List[StrategicAgent]) -> List[bool]:
        """
        Greedy approximation for optimal actions.
        
        Args:
            agents: List of strategic agents
            
        Returns:
            Approximately optimal action profile
        """
        # Sort agents by accuracy/energy ratio
        agent_scores = [
            (i, agent.camera.current_accuracy / agent.camera.energy_model.classification_cost)
            for i, agent in enumerate(agents)
            if agent.camera.can_classify()
        ]
        
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        actions = [False] * len(agents)
        current_welfare = 0.0
        
        # Greedily add agents
        for i, _ in agent_scores:
            actions[i] = True
            new_welfare = self.compute_social_welfare(agents, actions)
            
            if new_welfare > current_welfare:
                current_welfare = new_welfare
            else:
                actions[i] = False  # Remove if doesn't improve
                break
                
        return actions