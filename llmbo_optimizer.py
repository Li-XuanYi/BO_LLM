"""
LLMBO: Large Language Model Enhanced Bayesian Optimization
Implementation for lithium-ion battery charging optimization
Enhanced with sensitivity analysis and diverse sampling
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
        Generate initial points using LLM with enhanced diversity
        Uses temperature=0.9 based on research for maximum diversity
        
        Parameters:
        -----------
        n_initial : int
            Number of initial points to generate
            
        Returns:
        --------
        list : Initial parameter sets
        """
        print("🤖 LLM Warm Start: Generating physically plausible initial parameters...")
        print("   使用研究推荐配置: temperature=0.9, 高多样性策略")
        
        # Enhanced prompt with more specific guidance
        enhanced_prompt = f"""As an expert in electrochemistry and battery optimization, generate {n_initial * 2} diverse initial candidate parameter sets for lithium-ion battery charging optimization.

Parameter bounds:
- current1: {self.pbounds['current1']} A (initial charging current)
- charging_number: {self.pbounds['charging_number']} cycles (transition point)
- current2: {self.pbounds['current2']} A (final charging current)

Physical insights from battery research:
1. High current1 (5-6A) enables fast charging but risks lithium plating if held too long
2. Early transition (charging_number < 10) may not utilize high-current benefits
3. Late transition (charging_number > 20) increases temperature accumulation
4. Low current2 ensures safe completion near voltage limit

Generate DIVERSE exploration strategies:
- Strategy 1: Aggressive fast charging (high current1, medium transition)
- Strategy 2: Conservative approach (moderate current1, early transition)  
- Strategy 3: Two-stage optimization (high current1, late transition, low current2)
- Strategy 4: Balanced approach (medium values across all parameters)
- Strategy 5: Temperature-aware (moderate current1, adaptive transition)

Output {n_initial * 2} parameter sets as JSON array covering these strategies plus variations.
Ensure wide coverage of the parameter space for effective exploration.

Format:
[
  {{"current1": value, "charging_number": value, "current2": value}},
  ...
]"""
        
        # Use temperature=0.9 for maximum diversity (research recommendation)
        response = self.llm.generate_response(enhanced_prompt, temperature=0.9, max_tokens=2500)
        
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
            
            # Select diverse subset using clustering
            if len(valid_points) > n_initial:
                valid_points = self._select_diverse_points(valid_points, n_initial)
            
            # If insufficient valid points, supplement with Latin Hypercube Sampling
            while len(valid_points) < n_initial:
                lhs_point = self._latin_hypercube_sample()
                valid_points.append(lhs_point)
            
            print(f"✓ Generated {len(valid_points)} diverse initial points (LLM + LHS)")
            return valid_points[:n_initial]
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"⚠ LLM response parsing failed: {e}, using Latin Hypercube Sampling")
            return [self._latin_hypercube_sample() for _ in range(n_initial)]
    
    def _latin_hypercube_sample(self) -> Dict[str, float]:
        """Generate point using Latin Hypercube Sampling for better space coverage"""
        n_dims = len(self.pbounds)
        sample = np.random.uniform(0, 1, n_dims)
        
        point = {}
        for i, (key, bounds) in enumerate(sorted(self.pbounds.items())):
            # Add random jitter within the grid cell
            grid_value = (sample[i] + np.random.uniform(0, 1)) / n_dims
            point[key] = bounds[0] + grid_value * (bounds[1] - bounds[0])
        
        return point
    
    def _select_diverse_points(self, points: List[Dict[str, float]], n_select: int) -> List[Dict[str, float]]:
        """Select diverse subset of points using greedy maximin distance"""
        if len(points) <= n_select:
            return points
        
        # Convert to arrays
        points_array = np.array([[p[k] for k in sorted(self.pbounds.keys())] for p in points])
        
        # Normalize to [0, 1] for fair distance calculation
        bounds_array = np.array([self.pbounds[k] for k in sorted(self.pbounds.keys())])
        points_norm = (points_array - bounds_array[:, 0]) / (bounds_array[:, 1] - bounds_array[:, 0])
        
        # Greedy selection: start with random point, then add farthest points
        selected_indices = [np.random.randint(len(points))]
        
        for _ in range(n_select - 1):
            # Compute minimum distance to already selected points
            distances = []
            for i in range(len(points)):
                if i in selected_indices:
                    distances.append(-np.inf)
                else:
                    min_dist = min([np.linalg.norm(points_norm[i] - points_norm[j]) 
                                   for j in selected_indices])
                    distances.append(min_dist)
            
            # Select point with maximum minimum distance
            selected_indices.append(np.argmax(distances))
        
        return [points[i] for i in selected_indices]
    
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
    
    def _compute_parameter_sensitivity(self) -> Dict[str, float]:
        """
        Compute parameter sensitivity using finite differences
        Based on Morris sensitivity analysis methodology
        
        Returns:
        --------
        dict : Sensitivity score for each parameter
        """
        if len(self.y) < 3:
            return {k: 1.0 for k in self.pbounds.keys()}
        
        sensitivities = {}
        
        for param_name in self.pbounds.keys():
            # Compute elementary effects
            effects = []
            
            for i in range(1, len(self.X)):
                prev_point = self._array_to_dict(self.X[i-1])
                curr_point = self._array_to_dict(self.X[i])
                
                # Check if this parameter changed
                param_change = abs(curr_point[param_name] - prev_point[param_name])
                
                if param_change > 1e-6:  # Parameter changed
                    output_change = abs(self.y[i] - self.y[i-1])
                    effect = output_change / param_change
                    effects.append(effect)
            
            # Sensitivity is mean absolute effect
            if effects:
                sensitivities[param_name] = np.mean(effects)
            else:
                sensitivities[param_name] = 1.0
        
        # Normalize sensitivities
        total_sens = sum(sensitivities.values())
        if total_sens > 0:
            sensitivities = {k: v / total_sens for k, v in sensitivities.items()}
        
        return sensitivities
    
    def _probability_of_improvement(self, X: np.ndarray, xi: float = 0.01,
                                   param_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate Probability of Improvement (PI) acquisition function
        Research shows PI performs best for fast charging optimization
        
        Based on: Applied Energy 2022 paper - PI outperforms EI and LCB
        
        Parameters:
        -----------
        X : np.ndarray
            Points to evaluate
        xi : float
            Exploration parameter
        param_weights : dict
            Weights for each parameter (from sensitivity analysis)
            
        Returns:
        --------
        np.ndarray : PI values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if len(self.y) == 0:
            return np.zeros(len(X))
        
        mu_best = np.max(self.y)
        
        # Apply parameter sensitivity weights
        if param_weights is not None and self.iteration > 3:
            best_point = self.X[np.argmax(self.y)]
            for i in range(len(X)):
                weight_factor = 1.0
                for j, param_name in enumerate(sorted(self.pbounds.keys())):
                    param_sens = param_weights.get(param_name, 1.0)
                    param_diff = abs(X[i, j] - best_point[j])
                    weight_factor += param_sens * param_diff * 0.15
                sigma[i] *= weight_factor
        
        with np.errstate(divide='warn', invalid='warn'):
            Z = (mu - mu_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01, 
                            param_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate Expected Improvement with parameter-sensitive weighting
        
        Parameters:
        -----------
        X : np.ndarray
            Points to evaluate
        xi : float
            Exploration parameter
        param_weights : dict
            Weights for each parameter (from sensitivity analysis)
            
        Returns:
        --------
        np.ndarray : EI values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if len(self.y) == 0:
            return np.zeros(len(X))
        
        mu_best = np.max(self.y)
        
        # Apply parameter sensitivity weights
        if param_weights is not None and self.iteration > 3:
            best_point = self.X[np.argmax(self.y)]
            
            for i in range(len(X)):
                weight_factor = 1.0
                for j, param_name in enumerate(sorted(self.pbounds.keys())):
                    param_sens = param_weights.get(param_name, 1.0)
                    param_diff = abs(X[i, j] - best_point[j])
                    weight_factor += param_sens * param_diff * 0.1
                
                sigma[i] *= weight_factor
        
        with np.errstate(divide='warn', invalid='warn'):
            imp = mu - mu_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _suggest_next_point(self, n_candidates: int = 1000, use_sensitivity: bool = True,
                          acquisition_type: str = 'PI') -> Dict[str, float]:
        """
        Suggest next point to evaluate using acquisition function with sensitivity weighting
        Research recommendation: Use PI for fast charging optimization
        
        Parameters:
        -----------
        n_candidates : int
            Number of random candidates to evaluate
        use_sensitivity : bool
            Whether to use parameter sensitivity in acquisition
        acquisition_type : str
            Type of acquisition function ('PI', 'EI', 'LCB')
            Research shows PI performs best for fast charging
            
        Returns:
        --------
        dict : Next point to evaluate
        """
        # Compute parameter sensitivities
        param_weights = None
        if use_sensitivity and self.iteration > 3:
            param_weights = self._compute_parameter_sensitivity()
            print(f"   Parameter sensitivities: {{{', '.join([f'{k}: {v:.3f}' for k, v in param_weights.items()])}}}")
        
        # Generate candidates with focus on sensitive parameters
        candidates = []
        
        # 70% random candidates
        for _ in range(int(n_candidates * 0.7)):
            point = {
                key: np.random.uniform(bounds[0], bounds[1])
                for key, bounds in self.pbounds.items()
            }
            candidates.append(self._dict_to_array(point))
        
        # 30% candidates around best point (local exploitation)
        if len(self.X) > 0:
            best_point = self.X[np.argmax(self.y)]
            for _ in range(int(n_candidates * 0.3)):
                noise = []
                for j, param_name in enumerate(sorted(self.pbounds.keys())):
                    bounds = self.pbounds[param_name]
                    scale = (bounds[1] - bounds[0]) * 0.1
                    
                    # More noise for sensitive parameters (exploration)
                    if param_weights and param_name in param_weights:
                        scale *= (1 + param_weights[param_name])
                    
                    noise.append(np.random.normal(0, scale))
                
                perturbed = best_point + np.array(noise)
                
                # Clip to bounds
                for j, param_name in enumerate(sorted(self.pbounds.keys())):
                    bounds = self.pbounds[param_name]
                    perturbed[j] = np.clip(perturbed[j], bounds[0], bounds[1])
                
                candidates.append(perturbed)
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition function - use PI by default (research recommendation)
        if acquisition_type == 'PI':
            acq_values = self._probability_of_improvement(candidates, param_weights=param_weights)
            print(f"   使用PI acquisition function (论文推荐用于快充优化)")
        elif acquisition_type == 'LCB':
            acq_values = -self._lower_confidence_bound(candidates, beta=4.0)  # beta=4 from paper
            print(f"   使用LCB acquisition function (beta=4.0)")
        else:  # EI
            acq_values = self._expected_improvement(candidates, param_weights=param_weights)
            print(f"   使用EI acquisition function")
        
        # Select best candidate
        best_idx = np.argmax(acq_values)
        best_point = self._array_to_dict(candidates[best_idx])
        
        return best_point
    
    def _lower_confidence_bound(self, X: np.ndarray, beta: float = 4.0) -> np.ndarray:
        """
        Calculate Lower Confidence Bound
        Research shows beta=4.0 is optimal (Applied Energy 2022, Figure B1)
        
        Parameters:
        -----------
        X : np.ndarray
            Points to evaluate
        beta : float
            Exploration parameter (论文推荐: beta=4.0)
            
        Returns:
        --------
        np.ndarray : LCB values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu - beta * sigma
    
    def _llm_enhanced_surrogate(self) -> Dict[str, Any]:
        """
        Get LLM suggestions for surrogate model parameters
        Uses temperature=0.5 for precise physical understanding
        
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
        print("   使用研究推荐配置: temperature=0.5, 精确物理理解")
        
        best_idx = np.argmax(self.y)
        current_best = {
            'params': self._array_to_dict(self.X[best_idx]),
            'target': float(self.y[best_idx])  # Convert to native Python type
        }
        
        prompt = self.llm.surrogate_modeling_prompt(
            current_best=current_best,
            historical_data=self.history[-10:],
            parameter_info={}
        )
        
        # Use temperature=0.5 for precise analysis (research recommendation)
        response = self.llm.generate_response(prompt, temperature=0.5, max_tokens=2000)
        
        try:
            response_clean = response.strip()
            if '```json' in response_clean:
                response_clean = response_clean.split('```json')[1].split('```')[0]
            elif '```' in response_clean:
                response_clean = response_clean.split('```')[1].split('```')[0]
            
            config = json.loads(response_clean.strip())
            self.gamma = float(config.get('gamma', 1.0))
            print(f"✓ Updated coupling strength γ={self.gamma:.2f}")
            
            # Display length scales if available
            if 'kernel_length_scales' in config:
                scales = config['kernel_length_scales']
                print(f"   Length scales: {{{', '.join([f'{k}: {v:.2f}' for k, v in scales.items()])}}}")
            
            return config
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Using default surrogate configuration: {e}")
            return {'gamma': self.gamma, 'kernel_length_scales': {}}
    
    def _llm_candidate_sampling(self, n_candidates: int = 5) -> Dict[str, Any]:
        """
        Get LLM guidance for candidate sampling
        Uses adaptive temperature based on iteration: early=0.7, mid=0.6, late=0.5
        
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
        
        # Adaptive temperature based on iteration stage
        if self.iteration < 10:
            temp = 0.7
            stage = "早期探索"
        elif self.iteration < 25:
            temp = 0.6
            stage = "中期平衡"
        else:
            temp = 0.5
            stage = "后期利用"
        
        print(f"   使用研究推荐配置: temperature={temp}, {stage}")
        
        # Get current acquisition function suggestions
        candidates_list = []
        for _ in range(n_candidates):
            point = {
                key: np.random.uniform(bounds[0], bounds[1])
                for key, bounds in self.pbounds.items()
            }
            candidates_list.append(point)
        
        convergence_status = self._assess_convergence()
        
        # Compute parameter sensitivity for the prompt
        param_sensitivity = None
        if self.iteration > 3:
            param_sensitivity = self._compute_parameter_sensitivity()
        
        best_idx = np.argmax(self.y)
        prompt = self.llm.candidate_sampling_prompt(
            current_state={
                'iteration': self.iteration,
                'best_time': float(-self.y[best_idx]),
                'total_iterations': 30
            },
            acquisition_candidates=candidates_list,
            convergence_status=convergence_status,
            parameter_sensitivity=param_sensitivity
        )
        
        # Use adaptive temperature
        response = self.llm.generate_response(prompt, temperature=temp, max_tokens=1800)
        
        try:
            response_clean = response.strip()
            if '```json' in response_clean:
                response_clean = response_clean.split('```json')[1].split('```')[0]
            elif '```' in response_clean:
                response_clean = response_clean.split('```')[1].split('```')[0]
            
            strategy = json.loads(response_clean.strip())
            print(f"✓ Sampling strategy: {strategy.get('exploration_strategy', 'balanced')}")
            
            # Display weights if available
            if 'weights' in strategy:
                weights = strategy['weights']
                print(f"   Exploration weights: {{{', '.join([f'{k}: {v:.2f}' for k, v in weights.items()])}}}")
            
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
    