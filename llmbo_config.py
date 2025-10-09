
"""
LLM Configuration Optimizer
Based on research findings from LLMBO and fast charging papers
"""

from typing import Dict, Any


class LLMConfig:
    """
    LLM配置管理器 - 基于论文研究结果优化
    
    参考文献：
    1. Large language model-enhanced Bayesian optimization (Kuai et al.)
    2. Fast charging design via Bayesian optimization (Jiang et al., 2022)
    """
    
    # 基于论文结论的模型性能排序
    MODEL_RANKINGS = {
        'gpt-4o': {'rank': 1, 'accuracy': 'highest', 'cost': 'high'},
        'deepseek-r1': {'rank': 2, 'accuracy': 'highest', 'cost': 'medium'},
        'qwen-max': {'rank': 3, 'accuracy': 'high', 'cost': 'medium'},
        'qwen-plus': {'rank': 4, 'accuracy': 'high', 'cost': 'low'},
        'qwen-turbo': {'rank': 5, 'accuracy': 'medium', 'cost': 'lowest'}
    }
    
    @staticmethod
    def get_optimal_config(stage: str, iteration: int = 0) -> Dict[str, Any]:
        """
        根据优化阶段返回最优LLM配置
        
        Parameters:
        -----------
        stage : str
            优化阶段 ('warm_start', 'surrogate', 'sampling')
        iteration : int
            当前迭代次数
            
        Returns:
        --------
        dict : LLM配置字典
        """
        
        if stage == 'warm_start':
            # Warm Start阶段：需要高多样性和创造性
            return {
                'temperature': 0.9,  # 论文建议：高temperature增加多样性
                'max_tokens': 2500,  # 生成多个初始点需要更多tokens
                'top_p': 0.95,
                'reasoning': '需要生成多样化的初始策略，探索参数空间'
            }
            
        elif stage == 'surrogate':
            # Surrogate Modeling阶段：需要准确的物理理解
            return {
                'temperature': 0.5,  # 较低temperature提高精确性
                'max_tokens': 2000,
                'top_p': 0.9,
                'reasoning': '需要准确理解参数耦合关系，构建复合核函数'
            }
            
        elif stage == 'sampling':
            # Candidate Sampling阶段：平衡探索与利用
            # 早期探索，后期利用
            if iteration < 10:
                temp = 0.7  # 早期探索
            elif iteration < 25:
                temp = 0.6  # 中期平衡
            else:
                temp = 0.5  # 后期利用
                
            return {
                'temperature': temp,
                'max_tokens': 1800,
                'top_p': 0.9,
                'reasoning': f'迭代{iteration}：动态调整探索-利用平衡'
            }
        
        # 默认配置
        return {
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9,
            'reasoning': '默认配置'
        }
    
    @staticmethod
    def select_model(budget: str = 'medium', task_complexity: str = 'high') -> str:
        """
        根据预算和任务复杂度选择最优模型
        
        Parameters:
        -----------
        budget : str
            预算级别 ('low', 'medium', 'high')
        task_complexity : str
            任务复杂度 ('low', 'medium', 'high')
            
        Returns:
        --------
        str : 推荐的模型名称
        """
        
        if task_complexity == 'high':
            if budget == 'high':
                # 论文结论：GPT-4o在复杂任务中表现最佳
                return 'gpt-4o'
            elif budget == 'medium':
                # DeepSeek-R1性价比最高
                return 'deepseek-r1'
            else:
                return 'qwen-max'
        
        elif task_complexity == 'medium':
            if budget in ['high', 'medium']:
                return 'qwen-max'
            else:
                return 'qwen-plus'
        
        else:  # low complexity
            return 'qwen-turbo'
    
    @staticmethod
    def get_prompt_enhancement(stage: str) -> Dict[str, str]:
        """
        获取针对不同阶段的提示词增强策略
        基于论文图2和图3的提示词设计
        """
        
        enhancements = {
            'warm_start': {
                'prefix': """You are an expert electrochemist specializing in lithium-ion battery parameter optimization. 
Your task is to generate diverse, physically plausible initial parameter sets.""",
                
                'guidelines': """
Key principles from electrochemistry:
1. Higher current1 (5-6A) enables fast charging but increases lithium plating risk
2. Early transition (charging_number < 10) underutilizes high-current benefits
3. Late transition (charging_number > 20) causes excessive temperature accumulation
4. Lower current2 ensures safe voltage limit approach

Generate 10 parameter sets representing different strategies:
- Aggressive: High current1, moderate transition
- Conservative: Moderate current1, early transition  
- Balanced: Medium values across parameters
- Temperature-aware: Adaptive transitions based on thermal limits
- Voltage-limited: Early transition to prevent overvoltage
""",
                
                'output_format': 'JSON array with keys: current1, charging_number, current2'
            },
            
            'surrogate': {
                'prefix': """You are an expert in electrochemical modeling and Gaussian Process kernels.
Analyze parameter coupling effects in the pseudo-two-dimensional battery model.""",
                
                'guidelines': """
Coupling analysis framework (based on Equation 5):
k(θ, θ') = exp(-||θ - θ'||²/2l²) + γ · LLM_coupling(θ, θ' | S)

Critical coupling mechanisms:
1. Solid-phase diffusion vs. electrolyte volume fraction
   - Higher current1 → faster Li+ intercalation → concentration gradients
   - Affects voltage transients and lithium plating risk

2. Charging transition timing vs. temperature rise
   - Late transition → prolonged high-current phase → thermal accumulation
   - Early transition → reduced energy throughput → longer total time

3. Final current vs. voltage approach
   - CV-mode behavior near 4.2V limit
   - Exponential current decay: I = I₀ · exp(-α(V-V_ref))

Based on convergence trends, suggest:
- γ (coupling strength): 0.5-2.0 
- Length scales for each parameter
- Sensitive parameter identification
""",
                
                'output_format': 'JSON with gamma, kernel_length_scales, sensitive_params, rationale'
            },
            
            'sampling': {
                'prefix': """You are an expert in adaptive sampling for Bayesian optimization.
Analyze historical data to guide efficient parameter space exploration.""",
                
                'guidelines': """
Morris sensitivity analysis (μ* metric):
- Compute elementary effects: EE_j = Δf / Δθ_j
- Sensitivity μ_j* = mean(|EE_j|)

Adaptive strategy (based on iteration progress):
1. Early stage (iter < 10): 
   - High exploration weight (0.7-0.9)
   - Focus on sensitive parameters
   - Wide parameter range sampling

2. Mid stage (10 ≤ iter < 25):
   - Balanced exploration-exploitation (0.5-0.7)
   - Local search around promising regions
   - Adaptive length scales

3. Late stage (iter ≥ 25):
   - High exploitation weight (0.3-0.5)
   - Fine-tuning near optimum
   - Narrow search ranges

Output exploration weights for each parameter based on:
- Historical sensitivity (μ*)
- Convergence rate
- Current best value proximity
""",
                
                'output_format': 'JSON with weights, exploration_strategy, priority_regions, rationale'
            }
        }
        
        return enhancements.get(stage, {})


class AcquisitionFunctionConfig:
    """
    Acquisition Function配置
    基于Applied Energy 2022论文的实验结果
    """
    
    @staticmethod
    def get_pi_config() -> Dict[str, Any]:
        """
        Probability of Improvement (PI) 配置
        论文结论：PI在快充优化中表现最佳
        """
        return {
            'type': 'PI',
            'xi': 0.01,  # 探索参数
            'description': '论文实验显示PI收敛速度最快',
            'best_for': ['fast_charging', 'battery_optimization'],
            'advantages': [
                '收敛速度快',
                '对噪声鲁棒',
                '计算效率高'
            ]
        }
    
    @staticmethod
    def get_lcb_config() -> Dict[str, Any]:
        """
        Lower Confidence Bound (LCB) 配置
        """
        return {
            'type': 'LCB',
            'beta': 4.0,  # 论文图B1显示beta=4最优
            'description': 'LCB适合需要更多探索的场景',
            'best_for': ['high_dimensional', 'noisy_objectives'],
            'advantages': [
                '良好的exploration-exploitation平衡',
                '理论保证',
                'beta可调节'
            ]
        }
    
    @staticmethod
    def get_ei_config() -> Dict[str, Any]:
        """
        Expected Improvement (EI) 配置
        """
        return {
            'type': 'EI',
            'xi': 0.01,
            'description': 'EI是经典选择，但在快充中不如PI',
            'best_for': ['general_purpose', 'smooth_objectives'],
            'advantages': [
                '广泛使用',
                '数学性质好',
                '可解释性强'
            ]
        }
    
    @staticmethod
    def recommend_acquisition(problem_type: str = 'fast_charging') -> Dict[str, Any]:
        """
        根据问题类型推荐acquisition function
        """
        recommendations = {
            'fast_charging': AcquisitionFunctionConfig.get_pi_config(),
            'parameter_identification': AcquisitionFunctionConfig.get_ei_config(),
            'high_noise': AcquisitionFunctionConfig.get_lcb_config(),
            'general': AcquisitionFunctionConfig.get_ei_config()
        }
        
        return recommendations.get(problem_type, 
                                   AcquisitionFunctionConfig.get_ei_config())


# 使用示例
if __name__ == "__main__":
    config_manager = LLMConfig()
    
    print("="*60)
    print("LLM配置优化建议（基于论文研究）")
    print("="*60)
    
    # 1. 模型选择
    print("\n1. 模型选择建议：")
    print(f"   高性能需求: {config_manager.select_model('high', 'high')}")
    print(f"   性价比优先: {config_manager.select_model('medium', 'high')}")
    print(f"   预算受限: {config_manager.select_model('low', 'high')}")
    
    # 2. 各阶段配置
    print("\n2. 各阶段最优配置：")
    for stage in ['warm_start', 'surrogate', 'sampling']:
        config = config_manager.get_optimal_config(stage, iteration=5)
        print(f"\n   {stage}:")
        print(f"   - Temperature: {config['temperature']}")
        print(f"   - Max Tokens: {config['max_tokens']}")
        print(f"   - 原因: {config['reasoning']}")
    
    # 3. Acquisition Function建议
    print("\n3. Acquisition Function建议：")
    acq_config = AcquisitionFunctionConfig.recommend_acquisition('fast_charging')
    print(f"   推荐: {acq_config['type']}")
    print(f"   原因: {acq_config['description']}")
    print(f"   优势: {', '.join(acq_config['advantages'])}")
    
    print("\n" + "="*60)
    