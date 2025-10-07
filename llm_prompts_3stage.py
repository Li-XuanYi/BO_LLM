"""
三段式充电专用LLM提示词生成器
基于Applied Energy 307 (2022)和中文论文表4-3的物理洞察
"""


def get_warm_start_prompt_3stage(pbounds, n_initial=8):
    """
    生成三段式充电的warm start提示词
    """
    prompt = f"""You are an expert electrochemist specializing in lithium-ion battery fast charging optimization.

Your task is to generate {n_initial} diverse, physically plausible parameter sets for a THREE-STAGE charging protocol.

Parameter specifications:
- current1: {pbounds['current1']} A (first stage: high-current fast charging)
- charging_number1: {pbounds['charging_number1']} cycles (first transition point)
- current2: {pbounds['current2']} A (second stage: medium-current charging)
- charging_number2: {pbounds['charging_number2']} cycles (second transition point)
- current3: {pbounds['current3']} A (third stage: low-current trickle charging)

Physical insights from battery research:
1. **Stage 1 (High-Current Fast Charging)**:
   - High current1 (5.5-6.5A) enables rapid SOC increase
   - Early transition (charging_number1 < 10) wastes high-current capability
   - Optimal range: charging_number1 = 10-13 cycles

2. **Stage 2 (Medium-Current Balancing)**:
   - current2 (3.5-5.5A) balances speed vs. temperature/voltage stress
   - This stage prevents overheating while maintaining good charging speed
   - Duration: (charging_number2 - charging_number1) should be 8-12 cycles

3. **Stage 3 (Low-Current Completion)**:
   - current3 (2.0-3.5A) safely approaches voltage/SOC limit
   - Lower current reduces lithium plating risk near full charge
   - Crucial for long-term battery health

Reference protocols from literature:
- Protocol A (Table 4-3): 5.92A(10) → 4.92A(20) → 3.00A = 48 min, ΔT=6.69K
- Protocol B (Table 4-3): 5.34A(12) → 4.56A(20) → 3.00A = 52 min, ΔT=5.89K
- Applied Energy optimal: 6.0A(12) → 4.2A(22) → 3.0A ≈ 17.4 min

Generate {n_initial} DIVERSE parameter sets representing different strategies:

**Strategy Types to Cover:**
1. Aggressive Fast Charging: High current1 (>6.0A), early first transition, moderate current2
2. Conservative Balanced: Medium current1 (5.3-5.8A), gradual current reduction
3. Temperature-Optimized: Moderate currents, longer stage durations to minimize heat
4. Voltage-Aware: Early transitions to prevent voltage overshoot
5. Hybrid Approaches: Mix of above strategies

**Critical Constraints:**
- charging_number1 MUST be < charging_number2 (stages must be sequential)
- current1 > current2 > current3 (current should decrease through stages)
- Avoid extreme combinations that violate battery safety (e.g., 6.5A → 2.0A sudden drop)

Output format: JSON array with keys: current1, charging_number1, current2, charging_number2, current3

Example output structure:
[
  {{"current1": 6.0, "charging_number1": 12, "current2": 4.5, "charging_number2": 22, "current3": 3.0}},
  {{"current1": 5.5, "charging_number1": 10, "current2": 4.0, "charging_number2": 20, "current3": 2.5}},
  ...
]

Generate {n_initial} diverse, physically valid parameter sets now:"""
    
    return prompt


def get_candidate_sampling_prompt_3stage(pbounds, current_best, sensitivity_info=None):
    """
    生成三段式充电的candidate sampling提示词
    """
    
    # 计算当前最优的阶段持续时间
    stage1_duration = current_best.get('charging_number1', 0)
    stage2_duration = current_best.get('charging_number2', 0) - stage1_duration
    
    prompt = f"""You are optimizing a THREE-STAGE battery charging protocol using Bayesian optimization.

Current best parameters:
- Stage 1: {current_best.get('current1', 0):.2f}A for {stage1_duration:.0f} cycles
- Stage 2: {current_best.get('current2', 0):.2f}A for {stage2_duration:.0f} cycles  
- Stage 3: {current_best.get('current3', 0):.2f}A until completion

Parameter bounds:
- current1: {pbounds['current1']} A
- charging_number1: {pbounds['charging_number1']} cycles
- current2: {pbounds['current2']} A
- charging_number2: {pbounds['charging_number2']} cycles
- current3: {pbounds['current3']} A

Optimization insights:
1. **Current best performance analysis**:
   - Stage 1 duration ({stage1_duration:.0f} cycles): {'Too short' if stage1_duration < 10 else 'Reasonable' if stage1_duration < 14 else 'Possibly too long'}
   - Stage 2 duration ({stage2_duration:.0f} cycles): {'Too short' if stage2_duration < 8 else 'Good balance' if stage2_duration < 15 else 'Possibly too long'}
   - Current reduction: {(current_best.get('current1', 0) - current_best.get('current3', 0)):.2f}A total

2. **Physical coupling mechanisms**:
   - High current1 + late charging_number1 → excessive temperature rise
   - Large current1-to-current2 drop → voltage transients, potential instability
   - Early charging_number2 transition → underutilized stage 2 capacity
   
3. **Exploration strategies**:
   - If current1 is high (>6.0A), explore EARLIER charging_number1 to reduce thermal stress
   - If stage 2 duration is short, explore LATER charging_number2 for better balance
   - If current3 is low (<2.5A), consider HIGHER values to reduce total time

Generate 5 candidate parameter sets that:
- Explore promising regions near current best
- Test parameter coupling hypotheses
- Maintain physical feasibility (current1 > current2 > current3, charging_number1 < charging_number2)

Output as JSON array with keys: current1, charging_number1, current2, charging_number2, current3"""
    
    return prompt


def get_surrogate_kernel_prompt_3stage(current_best, historical_data):
    """
    生成三段式充电的surrogate kernel增强提示词
    """
    
    prompt = f"""You are designing a composite Gaussian Process kernel for THREE-STAGE battery charging optimization.

Current best result:
- Parameters: current1={current_best.get('current1', 0):.2f}A, charging_number1={current_best.get('charging_number1', 0):.0f}, 
             current2={current_best.get('current2', 0):.2f}A, charging_number2={current_best.get('charging_number2', 0):.0f},
             current3={current_best.get('current3', 0):.2f}A
- Charging time: {-current_best.get('target', 0) * 1.5 / 60:.1f} min

Kernel design task:
Build a composite kernel k(θ, θ') = k_RBF(θ, θ') + γ·k_coupling(θ, θ') where:
- k_RBF: standard RBF kernel for smooth interpolation
- k_coupling: electrochemical coupling kernel based on P2D model physics
- γ ∈ [0.5, 2.0]: coupling strength (higher = more physics influence)

Key coupling mechanisms in THREE-STAGE charging:

1. **Stage 1-to-2 Transition Coupling**:
   Correlation: f(current1, charging_number1, current2)
   - High current1 + late charging_number1 → temperature accumulation
   - Large ΔI = (current1 - current2) → voltage transients
   - Coupling weight: w₁ = |ΔI| / current1

2. **Stage 2-to-3 Transition Coupling**:
   Correlation: f(current2, charging_number2, current3)
   - Stage 2 duration = (charging_number2 - charging_number1)
   - Longer stage 2 → better thermal management
   - Coupling weight: w₂ = (charging_number2 - charging_number1) / 10

3. **Overall Current Profile Coupling**:
   Total current drop: ΔI_total = current1 - current3
   - Larger drop → more aggressive strategy → higher time variance
   - Coupling weight: w₃ = ΔI_total / 4.0

Recommend coupling strength γ based on current optimization state:
- Early stage (< 15 evaluations): γ = 1.2 (emphasize physics-guided exploration)
- Mid stage (15-30 evaluations): γ = 0.8 (balance physics and data)
- Late stage (> 30 evaluations): γ = 0.5 (trust empirical data more)

Current evaluation count: {len(historical_data)}

Provide:
1. Recommended γ value with justification
2. Length scale adjustments for sensitive parameters
3. Expected coupling patterns to watch for

Output as JSON with keys: gamma, length_scales, coupling_analysis"""
    
    return prompt


# 使用示例
if __name__ == "__main__":
    pbounds_3stage = {
        "current1": (5.0, 6.5),
        "charging_number1": (8, 15),
        "current2": (3.5, 5.5),
        "charging_number2": (15, 25),
        "current3": (2.0, 3.5)
    }
    
    # Warm start提示词
    warm_prompt = get_warm_start_prompt_3stage(pbounds_3stage, n_initial=8)
    print("="*70)
    print("Warm Start Prompt (三段式):")
    print("="*70)
    print(warm_prompt)
    
    # Candidate sampling提示词
    current_best = {
        'current1': 5.92,
        'charging_number1': 11,
        'current2': 4.80,
        'charging_number2': 21,
        'current3': 3.00,
        'target': -960
    }
    
    sampling_prompt = get_candidate_sampling_prompt_3stage(pbounds_3stage, current_best)
    print("\n" + "="*70)
    print("Candidate Sampling Prompt (三段式):")
    print("="*70)
    print(sampling_prompt)