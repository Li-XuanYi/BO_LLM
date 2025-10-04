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
        Enhanced based on manuscript1.pdf Figure 3 and Equation 5
        
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
        
        # Calculate improvement trend
        if len(historical_data) >= 5:
            recent_best = [max([h['target'] for h in historical_data[:i+1]]) 
                          for i in range(len(historical_data)-5, len(historical_data))]
            improvements = [recent_best[i] - recent_best[i-1] for i in range(1, len(recent_best))]
            avg_improvement = np.mean(improvements)
            trend = "converging" if avg_improvement < 0.1 else "exploring"
        else:
            trend = "early_stage"
        
        prompt = f"""As an expert in Gaussian Process kernels and electrochemical modeling, design a composite kernel that captures parameter coupling in the P2D battery model.

**Composite Kernel Framework (Equation 5):**
k(θ, θ') = exp(-||θ - θ'||²/2l²) + γ · LLM_coupling(θ, θ' | S)
           [Base RBF Kernel]      [Coupling Term]

**Current Optimization State:**
Best parameters found:
{json.dumps(current_best_clean, indent=2)}

Historical performance:
- Total evaluations: {len(historical_data)}
- Best charging time: {-max([h['target'] for h in historical_data]):.1f} seconds
- Optimization trend: {trend}
- Recent improvement rate: {"high" if trend == "exploring" else "low"}

**Electrochemical Coupling Mechanisms:**

1. **Solid-Phase Diffusion ↔ Current Density:**
   - High current1 → Steep Li+ concentration gradients
   - Affects intercalation uniformity and lithium plating risk
   - Coupling strength: High (voltage transients)

2. **Charging Transition ↔ Thermal Accumulation:**
   - Late transition (high charging_number) → Prolonged high-current phase
   - Temperature rises approximately T ≈ T₀ + k·I²·t
   - Coupling strength: Medium-High (thermal constraints)

3. **Final Current ↔ Voltage Approach:**
   - CV-mode near 4.2V: I = current2 · exp(-α(V-V_ref))
   - Exponential decay behavior
   - Coupling strength: Medium (constraint satisfaction)

4. **Current1 ↔ Charging_number Interaction:**
   - Synergistic effect on total charging time
   - High current1 + late transition → Fast but risky
   - Moderate current1 + early transition → Safe but slow
   - Coupling strength: High (primary optimization trade-off)

**Task: Determine Composite Kernel Parameters**

Based on the optimization state and electrochemical principles:

1. **Coupling Strength (γ):**
   - If converging (small recent improvements): **γ = 0.5-0.8** (reduce coupling, exploit)
   - If exploring (large variations): **γ = 1.2-2.0** (increase coupling, explore interactions)
   - Current recommendation based on {trend}: γ = ?

2. **Length Scales (l_j for each parameter):**
   - Sensitive parameters (large impact) → **smaller l** (higher resolution)
   - Insensitive parameters → **larger l** (smoother interpolation)
   
   Typical ranges:
   - current1: 0.3-0.8 (highly sensitive to voltage/temperature)
   - charging_number: 1.0-2.0 (medium sensitivity)
   - current2: 0.5-1.2 (sensitive near constraints)

3. **Identify Most Sensitive Parameters:**
   Based on voltage transient analysis and thermal accumulation patterns,
   which parameters show highest sensitivity (largest ∂f/∂θ)?

**Output JSON Format:**
{{
  "gamma": <float between 0.5-2.0>,
  "sensitive_params": ["param1", "param2"],
  "kernel_length_scales": {{
    "current1": <float>,
    "charging_number": <float>,
    "current2": <float>
  }},
  "coupling_mechanisms": ["mechanism1", "mechanism2"],
  "rationale": "<brief explanation of parameter choices based on current optimization state>"
}}

**Guidelines:**
- Higher γ when far from convergence (explore parameter interactions)
- Lower γ when near convergence (refine around optimum)
- Smaller length scales for parameters causing voltage/temperature violations
- Consider historical patterns: are there visible parameter correlations?

Provide ONLY the JSON output, no additional text."""
        
        return prompt
    
    def candidate_sampling_prompt(self, current_state: Dict[str, Any],
                                 acquisition_candidates: List[Dict[str, float]],
                                 convergence_status: str,
                                 parameter_sensitivity: Optional[Dict[str, float]] = None) -> str:
        """
        Generate prompt for candidate sampling
        Enhanced with Morris sensitivity analysis (manuscript1.pdf Figure 17)
        
        Parameters:
        -----------
        current_state : dict
            Current optimization state
        acquisition_candidates : list
            Candidate points from acquisition function
        convergence_status : str
            Convergence status description
        parameter_sensitivity : dict
            Morris μ* sensitivity metrics for each parameter
            
        Returns:
        --------
        str : Formatted prompt
        """
        iteration = current_state.get('iteration', 0)
        
        # Determine optimization stage
        if iteration < 10:
            stage = "Early Exploration"
            strategy_hint = "High exploration (0.7-0.9), focus on sensitive parameters"
        elif iteration < 25:
            stage = "Mid-stage Balance"
            strategy_hint = "Balanced exploration-exploitation (0.5-0.7)"
        else:
            stage = "Late Exploitation"
            strategy_hint = "High exploitation (0.3-0.5), fine-tuning near optimum"
        
        sens_info = ""
        if parameter_sensitivity:
            sens_info = f"""
**Morris Sensitivity Analysis (μ* metrics):**
{json.dumps(parameter_sensitivity, indent=2)}

Interpretation:
- μ* > 0.4: Highly sensitive (focus exploration here)
- 0.2 < μ* < 0.4: Moderate sensitivity  
- μ* < 0.2: Low sensitivity (can use larger step sizes)
"""
        
        prompt = f"""As an expert in adaptive sampling for Bayesian optimization, analyze the optimization state and guide efficient parameter space exploration using Morris sensitivity analysis.

**Current Optimization State:**
- Iteration: {iteration}/{current_state.get('total_iterations', 30)}
- Stage: **{stage}**
- Best charging time: {current_state.get('best_time', 'N/A')} seconds
- Convergence status: {convergence_status}
- Strategy recommendation: {strategy_hint}

{sens_info}

**Top Acquisition Function Candidates:**
{json.dumps(acquisition_candidates[:5], indent=2)}

**Morris Sensitivity Framework:**

Elementary Effect: EE_j = [f(θ + Δe_j) - f(θ)] / Δ

Sensitivity metric: μ_j* = mean(|EE_j|) over all evaluations

**Adaptive Sampling Strategy:**

1. **Early Stage (iter < 10)** - Current stage: {iteration < 10}
   - Exploration weight: 0.7-0.9
   - Focus on sensitive parameters (high μ*)
   - Wide search ranges to avoid premature convergence
   - Sample from regions NOT yet well-explored

2. **Mid Stage (10 ≤ iter < 25)** - Current stage: {10 <= iteration < 25}
   - Exploration weight: 0.5-0.7
   - Balance global exploration with local search
   - Prioritize regions around current best
   - Adaptive step sizes based on μ*

3. **Late Stage (iter ≥ 25)** - Current stage: {iteration >= 25}
   - Exploration weight: 0.3-0.5
   - Fine-tuning near optimum
   - Small step sizes for precise optimization
   - High exploitation around best point

**Task: Generate Adaptive Weights and Search Regions**

For each parameter, assign exploration weight (0-1) based on:
- Parameter sensitivity μ* (higher μ* → higher weight)
- Current convergence status
- Distance from current best value
- Historical fluctuation patterns

**Output JSON Format:**
{{
  "weights": {{
    "current1": <0-1, higher for more sensitive params>,
    "charging_number": <0-1>,
    "current2": <0-1>
  }},
  "exploration_strategy": "<focused|balanced|global>",
  "priority_regions": [
    {{
      "current1": [min, max],
      "charging_number": [min, max],
      "current2": [min, max],
      "rationale": "<why this region is promising>"
    }}
  ],
  "step_sizes": {{
    "current1": <Δθ based on sensitivity>,
    "charging_number": <Δθ>,
    "current2": <Δθ>
  }},
  "rationale": "<explanation of strategy based on stage and sensitivity>"
}}

**Guidelines:**
- Higher weights for parameters with μ* > 0.4
- {stage} requires {strategy_hint}
- Consider if current best is near parameter bounds
- Priority regions should cover high-potential areas based on acquisition values

Provide ONLY the JSON output, no additional text."""
        
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