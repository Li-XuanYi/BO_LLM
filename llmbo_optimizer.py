"""
LLMBO: Large Language Model Enhanced Bayesian Optimization
Implementation for lithium-ion battery charging optimization
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any, Callable, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import norm
from llm_interface import QwenLLMInterface


class LLMBOOptimizer:
    """
    LLMBO Optimizer combining Bayesian Optimization with LLM enhancements
    Simplified implementation that works with the existing bayes_opt structure
    """
    
    def __init__(self, objective_function: Callable, pbounds: Dict[str, Tuple[float, float]],
                 llm_interface: QwenLLMInterface, constraints: Dict[str, Any],
                 random_state: int = 1):
        """
        Initialize LLMBO Optimizer
        
        Parameters:
        -----------
        objective_function : callable
            Function to optimize (returns negative charging time)
        pbounds : dict
            Parameter bounds for optimization
        llm_interface : QwenLLMInterface
            LLM interface for prompts
        constraints : dict
            Operational constraints
        random_state : int
            Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.pbounds = pbounds
        self.llm = llm_interface
        self.constraints = constraints
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Storage for optimization history
        self.X = []  # Parameters
        self.y = []  # Objective values
        self.history = []
        self.gamma = 1.0  # Coupling strength
        self.iteration = 0
        
        # GP model
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        
    def _llm_warm_start(self, n_initial: int = 5) -> List[Dict[str, float]]:
        """
        Generate initial points using LLM
        
        Parameters:
        -----------
        n_initial : int
            Number of initial points to generate
            
        Returns:
        --------
        list : Initial parameter sets
        """
        print("🤖 LLM Warm Start: Generating physically plausible initial parameters...")
        
        prompt = self.llm.warm_start_prompt(self.pbounds, self.constraints)
        response = self.llm.generate_response(prompt, temperature=0.8)
        
        try:
            # Parse JSON response
            response_clean = response.strip()
            if '```json' in response_clean:
                response_clean = response_clean.split('```json')[1].split('```')[0]
            elif '```' in response_clean:
                response_clean = response_clean.split('```')[1].split('```')[0]
            
            initial_points = json.loads(response_clean.strip())
            
            # Validate and filter points
            valid_points = []
            for point in initial_points:
                if self._validate_point(point):
                    valid_points.append(point)
            
            # If insufficient valid points, supplement with random samples
            while len(valid_points) < n_initial:
                random_point = {
                    key: np.random.uniform(bounds[0], bounds[1])
                    for key, bounds in self.pbounds.items()
                }
                valid_points.append(random_point)
            
            print(f"✓ Generated {len(valid_points)} initial points")
            return valid_points[:n_initial]
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"⚠ LLM response parsing failed: {e}, using random initialization")
            return self._random_initial_points(n_initial)
    
    def _random_initial_points(self, n: int) -> List[Dict[str, float]]:
        """Generate random initial points as fallback"""
        return [
            {key: np.random.uniform(bounds[0], bounds[1])
             for key, bounds in self.pbounds.items()}
            for _ in range(n)
        ]
    
    def _validate_point(self, point: Dict[str, float]) -> bool:
        """Validate if point satisfies bounds"""
        for key, value in point.items():
            if key not in self.pbounds:
                return False
            bounds = self.pbounds[key]
            if value < bounds[0] or value > bounds[1]:
                return False
        return True
    
    def _dict_to_array(self, point: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array"""
        return np.array([point[key] for key in sorted(self.pbounds.keys())])
    
    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dict"""
        keys = sorted(self.pbounds.keys())
        return {key: float(val) for key, val in zip(keys, arr)}
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function
        
        Parameters:
        -----------
        X : np.ndarray
            Points to evaluate
        xi : float
            Exploration parameter
            
        Returns:
        --------
        np.ndarray : EI values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if len(self.y) == 0:
            return np.zeros(len(X))
        
        mu_best = np.max(self.y)
        
        with np.errstate(divide='warn', invalid='warn'):
            imp = mu - mu_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _suggest_next_point(self, n_candidates: int = 1000) -> Dict[str, float]:
        """
        Suggest next point to evaluate using acquisition function
        
        Parameters:
        -----------
        n_candidates : int
            Number of random candidates to evaluate
            
        Returns:
        --------
        dict : Next point to evaluate
        """
        # Generate random candidates
        candidates = []
        for _ in range(n_candidates):
            point = {
                key: np.random.uniform(bounds[0], bounds[1])
                for key, bounds in self.pbounds.items()
            }
            candidates.append(self._dict_to_array(point))
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition function
        ei_values = self._expected_improvement(candidates)
        
        # Select best candidate
        best_idx = np.argmax(ei_values)
        best_point = self._array_to_dict(candidates[best_idx])
        
        return best_point
    
    def _llm_enhanced_surrogate(self) -> Dict[str, Any]:
        """
        Get LLM suggestions for surrogate model parameters
        
        Returns:
        --------
        dict : Surrogate model configuration
        """
        if self.iteration < 3:
            return {
                'gamma': 1.0,
                'kernel_length_scales': {k: 1.0 for k in self.pbounds.keys()}
            }
        
        print("🧠 LLM Surrogate Modeling: Analyzing parameter coupling...")
        
        best_idx = np.argmax(self.y)
        current_best = {
            'params': self._array_to_dict(self.X[best_idx]),
            'target': self.y[best_idx]
        }
        
        prompt = self.llm.surrogate_modeling_prompt(
            current_best=current_best,
            historical_data=self.history[-10:],
            parameter_info={}
        )
        
        response = self.llm.generate_response(prompt, temperature=0.5)
        
        try:
            response_clean = response.strip()
            if '```json' in response_clean:
                response_clean = response_clean.split('```json')[1].split('```')[0]
            elif '```' in response_clean:
                response_clean = response_clean.split('```')[1].split('```')[0]
            
            config = json.loads(response_clean.strip())
            self.gamma = config.get('gamma', 1.0)
            print(f"✓ Updated coupling strength γ={self.gamma:.2f}")
            return config
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Using default surrogate configuration: {e}")
            return {'gamma': self.gamma, 'kernel_length_scales': {}}
    
    def _llm_candidate_sampling(self, n_candidates: int = 5) -> Dict[str, Any]:
        """
        Get LLM guidance for candidate sampling
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidates to consider
            
        Returns:
        --------
        dict : Sampling strategy
        """
        if self.iteration < 2:
            return {'weights': {k: 1.0 for k in self.pbounds.keys()}}
        
        print("🎯 LLM Candidate Sampling: Identifying high-potential regions...")
        
        # Get current acquisition function suggestions
        candidates_list = []
        for _ in range(n_candidates):
            point = {
                key: np.random.uniform(bounds[0], bounds[1])
                for key, bounds in self.pbounds.items()
            }
            candidates_list.append(point)
        
        convergence_status = self._assess_convergence()
        
        best_idx = np.argmax(self.y)
        prompt = self.llm.candidate_sampling_prompt(
            current_state={
                'iteration': self.iteration,
                'best_time': -self.y[best_idx]
            },
            acquisition_candidates=candidates_list,
            convergence_status=convergence_status
        )
        
        response = self.llm.generate_response(prompt, temperature=0.6)
        
        try:
            response_clean = response.strip()
            if '```json' in response_clean:
                response_clean = response_clean.split('```json')[1].split('```')[0]
            elif '```' in response_clean:
                response_clean = response_clean.split('```')[1].split('```')[0]
            
            strategy = json.loads(response_clean.strip())
            print(f"✓ Sampling strategy: {strategy.get('exploration_strategy', 'balanced')}")
            return strategy
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Using default sampling strategy: {e}")
            return {'weights': {k: 1.0 for k in self.pbounds.keys()}}
    
    def _assess_convergence(self) -> str:
        """Assess convergence status"""
        if len(self.y) < 5:
            return "early_exploration"
        
        recent_best = [max(self.y[:i+1]) for i in range(len(self.y)-4, len(self.y))]
        improvements = [recent_best[i] - recent_best[i-1] for i in range(1, len(recent_best))]
        
        avg_improvement = np.mean(improvements)
        if avg_improvement < 0.1:
            return "converging"
        elif avg_improvement < 1.0:
            return "slow_progress"
        else:
            return "exploring"
    
    def optimize(self, init_points: int = 5, n_iter: int = 30) -> Dict[str, Any]:
        """
        Run LLMBO optimization
        
        Parameters:
        -----------
        init_points : int
            Number of initial points (will use LLM warm start)
        n_iter : int
            Number of optimization iterations
            
        Returns:
        --------
        dict : Optimization results
        """
        print("=" * 60)
        print("LLMBO: Large Language Model Enhanced Bayesian Optimization")
        print("=" * 60)
        
        # Phase 1: LLM-based warm start
        initial_points = self._llm_warm_start(init_points)
        
        # Evaluate initial points
        for point in initial_points:
            target_value = self.objective_function(**point)
            x_array = self._dict_to_array(point)
            
            self.X.append(x_array)
            self.y.append(target_value)
            self.history.append({
                'params': point,
                'target': target_value,
                'time': -target_value
            })
            print(f"Initial: {point} -> Time: {-target_value:.1f}s")
        
        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Phase 2: LLM-enhanced optimization loop
        for i in range(n_iter):
            self.iteration = i + 1
            print(f"\n--- Iteration {self.iteration}/{n_iter} ---")
            
            # Fit GP model
            self.gp.fit(self.X, self.y)
            
            # Get LLM guidance for surrogate model
            surrogate_config = self._llm_enhanced_surrogate()
            
            # Get LLM guidance for candidate sampling
            sampling_strategy = self._llm_candidate_sampling()
            
            # Suggest next point
            next_point = self._suggest_next_point()
            
            # Evaluate next point
            target_value = self.objective_function(**next_point)
            x_array = self._dict_to_array(next_point)
            
            # Update history
            self.X = np.vstack([self.X, x_array])
            self.y = np.append(self.y, target_value)
            self.history.append({
                'params': next_point,
                'target': target_value,
                'time': -target_value
            })
            
            print(f"Next: {next_point} -> Time: {-target_value:.1f}s")
            print(f"Best so far: {-np.max(self.y):.1f}s")
        
        # Final results
        best_idx = np.argmax(self.y)
        best_params = self._array_to_dict(self.X[best_idx])
        best_time = -self.y[best_idx]
        
        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print(f"Best charging time: {best_time:.1f} seconds")
        print(f"Optimal parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value:.2f}")
        print("=" * 60)
        
        return {
            'best_params': best_params,
            'best_time': best_time,
            'history': self.history,
            'X': self.X,
            'y': self.y
        }