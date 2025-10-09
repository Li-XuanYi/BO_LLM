"""
LLM Interface for Qwen API Integration
Provides interface to interact with Qwen LLM for battery optimization
"""

import requests
import json
import numpy as np
from typing import Dict, List, Any, Optional

class QwenLLMInterface:
    """Interface for Qwen LLM API calls"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus"):
        """
        Initialize Qwen LLM Interface
        
        Parameters:
        -----------
        api_key : str
            API key for Qwen
        base_url : str
            Base URL for Qwen API
        model : str
            Model name (default: qwen-plus)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 2000) -> str:
        """
        Generate response from LLM
        
        Parameters:
        -----------
        prompt : str
            Input prompt for LLM
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens in response
            
        Returns:
        --------
        str : Generated response
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in electrochemistry and battery parameter optimization."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error calling Qwen API: {e}")
            return ""
    
    def warm_start_prompt(self, pbounds: Dict[str, tuple], 
                         constraints: Dict[str, Any]) -> str:
        """
        Generate prompt for warm start initialization
        
        Parameters:
        -----------
        pbounds : dict
            Parameter bounds
        constraints : dict
            Constraint specifications
            
        Returns:
        --------
        str : Formatted prompt
        """
        prompt = f"""As an expert in electrochemistry and battery optimization, generate initial candidate parameter sets for lithium-ion battery charging optimization using the pseudo-two-dimensional model.

Parameter specifications:
{json.dumps(pbounds, indent=2)}

Operational constraints:
- Maximum voltage: {constraints.get('voltage_max', 4.2)} V
- Maximum temperature: {constraints.get('temp_max', 309)} K
- Current range: {pbounds.get('current1', (3, 6))} A for first stage, {pbounds.get('current2', (1, 3))} A for second stage
- Charging transitions: {pbounds.get('charging_number', (5, 25))} cycles

Physical considerations:
1. Higher initial current (current1) accelerates charging but increases temperature and voltage stress
2. Transition point (charging_number) should balance fast charging with constraint satisfaction
3. Final current (current2) should allow safe completion near voltage limit

Generate 5 physically plausible parameter sets as JSON array with keys: current1, charging_number, current2.
Each set should satisfy physical constraints and represent diverse exploration points.

Output format:
[
  {{"current1": value1, "charging_number": value2, "current2": value3}},
  ...
]

Only output the JSON array, no additional text."""
        
        return prompt
    
    def surrogate_modeling_prompt(self, current_best: Dict[str, float],
                                 historical_data: List[Dict[str, Any]],
                                 parameter_info: Dict[str, Any]) -> str:
        """
        Generate prompt for surrogate model enhancement
        
        Parameters:
        -----------
        current_best : dict
            Current best parameters
        historical_data : list
            Historical evaluation data
        parameter_info : dict
            Parameter sensitivity information
            
        Returns:
        --------
        str : Formatted prompt
        """
        # Convert numpy types to native Python types for JSON serialization
        current_best_clean = self._convert_to_native_types(current_best)
        historical_data_clean = self._convert_to_native_types(historical_data)
        
        prompt = f"""Design a Bayesian optimization surrogate model that explicitly captures electrochemical interactions in the pseudo-two-dimensional battery model.

Current best parameters:
{json.dumps(current_best_clean, indent=2)}

Historical evaluation summary:
- Total evaluations: {len(historical_data)}
- Best charging time: {min([d['time'] for d in historical_data])} seconds
- Parameter ranges explored: {self._summarize_ranges(historical_data)}

Electrochemical coupling considerations:
1. Higher current1 causes voltage transients due to concentration gradients
2. Transition timing (charging_number) affects lithium plating risk
3. Final current2 interacts with temperature rise from earlier stages

Construct a composite kernel function combining:
- Base RBF kernel for smooth interpolation
- Coupling term capturing synergistic effects between solid-phase diffusion and electrolyte volume fraction using gradient correlation analysis
- Dynamic length scale adjusting for sensitive parameters causing voltage transients

Provide coupling strength parameter (gamma) between 0.5-2.0 based on convergence trends. If recent improvement is high, increase gamma to strengthen coupling effects; otherwise decrease to prevent overfitting.

Output as JSON:
{{
  "gamma": <value>,
  "sensitive_params": [<list of parameter names>],
  "kernel_length_scales": {{"current1": <value>, "charging_number": <value>, "current2": <value>}},
  "rationale": "<brief explanation>"
}}"""
        
        return prompt
    
    def candidate_sampling_prompt(self, current_state: Dict[str, Any],
                                 acquisition_candidates: List[Dict[str, float]],
                                 convergence_status: str) -> str:
        """
        Generate prompt for candidate sampling
        
        Parameters:
        -----------
        current_state : dict
            Current optimization state
        acquisition_candidates : list
            Candidate points from acquisition function
        convergence_status : str
            Convergence status description
            
        Returns:
        --------
        str : Formatted prompt
        """
        prompt = f"""Analyze historical optimization data from P2D model simulations to adaptively guide parameter exploration, identifying highly sensitive parameters for voltage and dynamically adjusting their search ranges while prioritizing exploration of high-potential regions based on sensitivity analysis and maintaining global random sampling to balance exploration and exploitation.

Current optimization state:
- Iteration: {current_state.get('iteration', 0)}
- Best charging time: {current_state.get('best_time', 'N/A')} seconds
- Convergence status: {convergence_status}

Top candidates from acquisition function:
{json.dumps(acquisition_candidates[:5], indent=2)}

Task: Generate exploration weights for each parameter based on sensitivity to objective function. Parameters showing high sensitivity (large fluctuations in charging time) should receive higher weights for focused exploration.

Output as JSON:
{{
  "weights": {{"current1": <0-1>, "charging_number": <0-1>, "current2": <0-1>}},
  "exploration_strategy": "<focused/balanced/global>",
  "priority_regions": [
    {{"current1": [min, max], "charging_number": [min, max], "current2": [min, max]}}
  ],
  "rationale": "<explanation>"
}}"""
        
        return prompt
    
    def _summarize_ranges(self, historical_data: List[Dict[str, Any]]) -> Dict[str, tuple]:
        """Summarize parameter ranges from historical data"""
        if not historical_data:
            return {}
        
        params = list(historical_data[0]['params'].keys())
        ranges = {}
        for param in params:
            values = [d['params'][param] for d in historical_data]
            ranges[param] = (min(values), max(values))
        
        return ranges
    
    def _convert_to_native_types(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization
        
        Parameters:
        -----------
        obj : any
            Object to convert
            
        Returns:
        --------
        any : Object with native Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj