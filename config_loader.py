
"""
配置加载器
支持YAML配置文件和命令行覆盖
"""

import yaml
import os
from typing import Dict, Any, Optional
import argparse


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化配置
        
        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ 加载配置文件: {self.config_path}")
            return config
        else:
            print(f"⚠ 配置文件不存在: {self.config_path}，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'llm': {
                'api_key': "sk-84ac2d321cf444e799ddc9db79c02e92",
                'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
                'model': 'qwen-plus',
                'temperature': {
                    'warm_start': 0.9,
                    'surrogate': 0.5,
                    'sampling_early': 0.7,
                    'sampling_mid': 0.6,
                    'sampling_late': 0.5
                },
                'max_tokens': {
                    'warm_start': 2500,
                    'surrogate': 2000,
                    'sampling': 1800
                }
            },
            'bayesian_optimization': {
                'init_points': 5,
                'n_iter': 25,
                'acquisition_type': 'PI',
                'xi': 0.01,
                'beta': 4.0,
                'n_candidates': 1000,
                'use_sensitivity': True,
                'random_state': 1
            },
            'parameter_bounds': {
                'current1': {'min': 3.0, 'max': 6.0},
                'charging_number': {'min': 5, 'max': 25},
                'current2': {'min': 1.0, 'max': 3.0}
            },
            'constraints': {
                'voltage_max': 4.2,
                'temp_max': 309,
                'target_soc': 0.8
            },
            'visualization': {
                'dpi': 300,
                'current_smooth_window': 5
            },
            'experiment': {
                'mode': 'llmbo',
                'save_results': True
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）
        
        Parameters:
        -----------
        key_path : str
            配置键路径，如 'llm.model' 或 'bayesian_optimization.init_points'
        default : any
            默认值
            
        Returns:
        --------
        any : 配置值
        
        Example:
        --------
        >>> config.get('llm.model')
        'qwen-plus'
        >>> config.get('bayesian_optimization.n_iter')
        25
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_pbounds(self) -> Dict[str, tuple]:
        """获取参数边界（转换为元组格式）"""
        bounds = self.get('parameter_bounds', {})
        return {
            key: (val['min'], val['max'])
            for key, val in bounds.items()
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return {
            'api_key': self.get('llm.api_key'),
            'base_url': self.get('llm.base_url'),
            'model': self.get('llm.model'),
            'temperature': self.get('llm.temperature'),
            'max_tokens': self.get('llm.max_tokens')
        }
    
    def get_bo_config(self) -> Dict[str, Any]:
        """获取贝叶斯优化配置"""
        return self.get('bayesian_optimization', {})
    
    def get_constraints(self) -> Dict[str, Any]:
        """获取约束条件"""
        return self.get('constraints', {})
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "="*60)
        print("当前优化配置")
        print("="*60)
        
        print("\n📱 LLM配置:")
        print(f"  模型: {self.get('llm.model')}")
        print(f"  Temperature: Warm={self.get('llm.temperature.warm_start')}, "
              f"Surrogate={self.get('llm.temperature.surrogate')}, "
              f"Sampling={self.get('llm.temperature.sampling_early')}-{self.get('llm.temperature.sampling_late')}")
        
        print("\n🔧 优化配置:")
        print(f"  初始点: {self.get('bayesian_optimization.init_points')}")
        print(f"  迭代次数: {self.get('bayesian_optimization.n_iter')}")
        print(f"  Acquisition: {self.get('bayesian_optimization.acquisition_type')}")
        print(f"  使用敏感性分析: {self.get('bayesian_optimization.use_sensitivity')}")
        
        print("\n📊 参数边界:")
        for param, bounds in self.get_pbounds().items():
            print(f"  {param}: [{bounds[0]}, {bounds[1]}]")
        
        print("\n⚡ 约束条件:")
        constraints = self.get_constraints()
        print(f"  最大电压: {constraints.get('voltage_max')} V")
        print(f"  最大温度: {constraints.get('temp_max')} K")
        print(f"  目标SOC: {constraints.get('target_soc')}")
        
        print("\n" + "="*60 + "\n")
    
    def override_from_args(self, args: argparse.Namespace):
        """从命令行参数覆盖配置"""
        if hasattr(args, 'model') and args.model:
            self.config['llm']['model'] = args.model
            print(f"✓ 覆盖配置: LLM模型 = {args.model}")
        
        if hasattr(args, 'init_points') and args.init_points:
            self.config['bayesian_optimization']['init_points'] = args.init_points
            print(f"✓ 覆盖配置: 初始点 = {args.init_points}")
        
        if hasattr(args, 'n_iter') and args.n_iter:
            self.config['bayesian_optimization']['n_iter'] = args.n_iter
            print(f"✓ 覆盖配置: 迭代次数 = {args.n_iter}")
        
        if hasattr(args, 'acquisition') and args.acquisition:
            self.config['bayesian_optimization']['acquisition_type'] = args.acquisition
            print(f"✓ 覆盖配置: Acquisition = {args.acquisition}")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LLMBO - 基于LLM增强的贝叶斯优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行LLMBO
  python main_LLMBO_v2.py
  
  # 只运行标准BO
  python main_LLMBO_v2.py --mode bo
  
  # 使用不同模型
  python main_LLMBO_v2.py --model qwen-max
  
  # 自定义优化参数
  python main_LLMBO_v2.py --init-points 10 --n-iter 40
  
  # 使用PI acquisition function
  python main_LLMBO_v2.py --acquisition PI
  
  # 指定配置文件
  python main_LLMBO_v2.py --config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    
    parser.add_argument('--mode', type=str, choices=['bo', 'llmbo', 'both'],
                       help='运行模式: bo (标准BO), llmbo (LLMBO), both (对比)')
    
    parser.add_argument('--model', type=str,
                       choices=['qwen-turbo', 'qwen-plus', 'qwen-max', 'gpt-4o', 'deepseek-r1'],
                       help='LLM模型选择')
    
    parser.add_argument('--init-points', type=int,
                       help='初始采样点数')
    
    parser.add_argument('--n-iter', type=int,
                       help='优化迭代次数')
    
    parser.add_argument('--acquisition', type=str, choices=['PI', 'EI', 'LCB'],
                       help='Acquisition function类型')
    
    parser.add_argument('--no-sensitivity', action='store_true',
                       help='禁用参数敏感性分析')
    
    parser.add_argument('--seed', type=int,
                       help='随机种子')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def load_config_with_args() -> Config:
    """
    加载配置并应用命令行参数
    
    Returns:
    --------
    Config : 配置对象
    """
    args = parse_arguments()
    config = Config(args.config)
    
    # 应用命令行覆盖
    config.override_from_args(args)
    
    # 应用额外参数
    if args.no_sensitivity:
        config.config['bayesian_optimization']['use_sensitivity'] = False
        print("✓ 禁用参数敏感性分析")
    
    if args.seed is not None:
        config.config['bayesian_optimization']['random_state'] = args.seed
        print(f"✓ 设置随机种子 = {args.seed}")
    
    if args.verbose:
        config.config['logging']['level'] = 'DEBUG'
        config.config['debug']['verbose'] = True
        print("✓ 启用详细输出模式")
    
    # 如果指定了mode，覆盖配置
    if args.mode:
        config.config['experiment']['mode'] = args.mode
    
    return config


if __name__ == "__main__":
    # 测试配置加载
    config = load_config_with_args()
    config.print_config()
    
    print("\n示例访问:")
    print(f"LLM模型: {config.get('llm.model')}")
    print(f"参数边界: {config.get_pbounds()}")
    print(f"Acquisition类型: {config.get('bayesian_optimization.acquisition_type')}")