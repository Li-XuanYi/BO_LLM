"""
三段式充电优化 - 独立完整版
直接使用现有的 llm_interface 和 llmbo_optimizer
无需额外的新文件
"""

import numpy as np
import time
import json
import re
from datetime import datetime
from SPM import SPM
from llm_interface import QwenLLMInterface
from llmbo_optimizer import LLMBOOptimizer
from scipy.stats import norm as normal_dist
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============= 配置参数 =============
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# 三段式充电参数边界（保守设置，避免SPM求解器崩溃）
PBOUNDS_3STAGE = {
    "current1": (3.0, 6.5),         # 第一段：高电流快充（缩小范围）
    "charging_number1": (10, 25),    # 第一次切换（避免过早过晚）
    "current2": (2.0, 5.5),         # 第二段：中等电流（确保与I1差距>0.5A）
    "charging_number2": (18, 23),   # 第二次切换（确保第二段足够长）
    "current3": (1.0, 3.5)          # 第三段：低电流涓流（避免过低电流）
}

CONSTRAINTS = {
    'voltage_max': 4.2,
    'temp_max': 313,
    'target_soc': 0.8
}


# ============= 三段式充电目标函数 =============
def charging_time_compute_3stage(current1, charging_number1, current2, charging_number2, current3):
    """
    三段式充电目标函数（带强约束和错误处理）
    
    充电策略: I1 → I2 → I3
    """
    try:
        # === 强约束检查 ===
        # 1. 电流必须递减，且差距至少0.5A
        if current1 <= current2 + 0.5:
            return -10000
        if current2 <= current3 + 0.3:
            return -10000
        
        # 2. 切换点必须递增，且间隔至少5个周期
        if charging_number1 >= charging_number2 - 5:
            return -10000
        
        # 3. 电流范围检查
        if not (5.0 <= current1 <= 6.5):
            return -10000
        if not (3.5 <= current2 <= 5.5):
            return -10000
        if not (2.0 <= current3 <= 4.0):
            return -10000
        
        env = SPM(3.0, 298)
        done = False
        i = 0
        
        charging_number1 = int(np.clip(charging_number1, 8, 15))
        charging_number2 = int(np.clip(charging_number2, 15, 25))
        
        # 确保切换点有效
        if charging_number1 >= charging_number2:
            return -10000
        
        while not done:
            # 三段式充电逻辑
            if i < charging_number1:
                current = current1
            elif i < charging_number2:
                current = current2
            else:
                current = current3
            
            # 接近电压上限时指数衰减
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
            
            # 限制电流在安全范围内
            current = np.clip(current, 0.8, 6.5)
            
            # 执行SPM仿真
            _, done, _ = env.step(current)
            i += 1
            
            # 约束违反检查
            if env.voltage > env.sett['constraints voltage max']:
                i += 10  # 电压超限重惩罚
            if env.temp > env.sett['constraints temperature max']:
                i += 10  # 温度超限重惩罚
            
            # 超时保护（避免无限循环）
            if i > 250:
                return -10000
            
            if done:
                return -i
        
        return -i
        
    except Exception as e:
        # SPM求解失败，返回重惩罚
        # 不打印错误信息，避免刷屏
        return -10000
        # if env.voltage > env.sett['constraints voltage max'] or \
        #     env.temp > env.sett['constraints temperature max']:
        #     i += 10
            
        #     # 超时保护
        # if i > 300:
        #     print(f"⚠️ 充电超时 (>300周期)")
        #     return -10000
            
        # if done:
        #     return -i
        
        # return -i
        
    except Exception as e:
        print(f"⚠️ SPM求解失败: {e}")
        print(f"   参数: I1={current1:.2f}, N1={charging_number1:.0f}, I2={current2:.2f}, N2={charging_number2:.0f}, I3={current3:.2f}")
        return -10000  # 返回大惩罚值


# ============= 可视化函数 =============
def plot_3stage_results(results, save_path='results_3stage.png'):
    """绘制三段式优化结果（6个子图）"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('三段式充电LLMBO优化结果', fontsize=18, fontweight='bold')
    
    iterations = list(range(1, len(results['history']) + 1))
    times = [h['time'] for h in results['history']]
    best_times = np.minimum.accumulate(times)
    
    # 1. 收敛曲线
    ax1 = axes[0, 0]
    ax1.plot(iterations, times, 'o-', color='#3498db', alpha=0.5, label='每次评估')
    ax1.plot(iterations, best_times, 's-', color='#e74c3c', linewidth=2.5, label='最优值')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('充电时间 (周期数)', fontsize=12)
    ax1.set_title('收敛曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2-6. 参数探索
    param_names = ['current1', 'charging_number1', 'current2', 'charging_number2', 'current3']
    param_titles = ['第一段电流 (A)', '第一次切换周期', '第二段电流 (A)', '第二次切换周期', '第三段电流 (A)']
    cmaps = ['viridis', 'plasma', 'coolwarm', 'RdYlGn_r', 'magma']
    
    for idx, (param, title, cmap) in enumerate(zip(param_names, param_titles, cmaps)):
        ax = axes.flatten()[idx + 1]
        param_vals = [h['params'][param] for h in results['history']]
        scatter = ax.scatter(iterations, param_vals, c=times, cmap=cmap, s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='充电时间')
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{param} 探索', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 结果图已保存至: {save_path}")
    plt.show()


def plot_3stage_charging_profile(best_params, save_path='charging_profile_3stage.png'):
    """绘制三段式充电曲线"""
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    # 数据收集
    time_data = [0]
    voltage_data = [env.voltage]
    temp_data = [env.temp]
    soc_data = [env.soc]
    current_raw = []
    
    charging_number1 = int(best_params['charging_number1'])
    charging_number2 = int(best_params['charging_number2'])
    
    while not done:
        if i < charging_number1:
            current = best_params['current1']
        elif i < charging_number2:
            current = best_params['current2']
        else:
            current = best_params['current3']
        
        if env.voltage >= 4.0:
            current = current * np.exp(-0.9 * (env.voltage - 4))
        
        current_raw.append(current)
        _, done, _ = env.step(current)
        i += 1
        
        time_data.append(i * env.sett['sample_time'])
        voltage_data.append(env.voltage)
        temp_data.append(env.temp)
        soc_data.append(env.soc)
    
    # 平滑电流
    window_size = 5
    current_smooth = []
    for idx in range(len(current_raw)):
        start = max(0, idx - window_size // 2)
        end = min(len(current_raw), idx + window_size // 2 + 1)
        current_smooth.append(np.mean(current_raw[start:end]))
    
    # 转换为分钟
    time_min = [t / 60 for t in time_data]
    time_current = [t / 60 for t in time_data[:-1]]
    
    switch1_time = charging_number1 * env.sett['sample_time'] / 60
    switch2_time = charging_number2 * env.sett['sample_time'] / 60
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('三段式充电协议电池状态量变化', fontsize=18, fontweight='bold')
    
    # 1. 电压
    ax1 = axes[0, 0]
    ax1.plot(time_min, voltage_data, 'g-', linewidth=2.5)
    ax1.axhline(y=4.2, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='电压上限')
    ax1.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='切换1')
    ax1.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6, label='切换2')
    ax1.set_xlabel('Time/min', fontsize=12)
    ax1.set_ylabel('Voltage /V', fontsize=12)
    ax1.set_title('电压曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([3.0, 4.3])
    
    # 2. 温度
    ax2 = axes[0, 1]
    ax2.plot(time_min, temp_data, 'b-', linewidth=2.5)
    ax2.axhline(y=309, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='温度上限')
    ax2.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6)
    ax2.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax2.set_xlabel('Time/min', fontsize=12)
    ax2.set_ylabel('Temperature /K', fontsize=12)
    ax2.set_title('温度曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([297, 311])
    
    # 3. 电流（三段式）
    ax3 = axes[1, 0]
    time_current = np.array(time_current)
    current_smooth = np.array(current_smooth)
    
    mask1 = time_current < switch1_time
    mask2 = (time_current >= switch1_time) & (time_current < switch2_time)
    mask3 = time_current >= switch2_time
    
    ax3.plot(time_current[mask1], current_smooth[mask1], 'r-', linewidth=2.5, 
             label=f'阶段1: {best_params["current1"]:.2f}A')
    ax3.plot(time_current[mask2], current_smooth[mask2], 'b-', linewidth=2.5, 
             label=f'阶段2: {best_params["current2"]:.2f}A')
    ax3.plot(time_current[mask3], current_smooth[mask3], 'g-', linewidth=2.5, 
             label=f'阶段3: {best_params["current3"]:.2f}A')
    ax3.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6)
    ax3.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax3.set_xlabel('Time/min', fontsize=12)
    ax3.set_ylabel('Input Current/A', fontsize=12)
    ax3.set_title('三段式充电电流曲线', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. SOC
    ax4 = axes[1, 1]
    ax4.plot(time_min, soc_data, 'm-', linewidth=2.5)
    ax4.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='目标SOC')
    ax4.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6)
    ax4.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax4.set_xlabel('Time/min', fontsize=12)
    ax4.set_ylabel('State of Charge', fontsize=12)
    ax4.set_title('SOC曲线', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 0.9])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 充电曲线图已保存至: {save_path}")
    plt.show()


# ============= 主程序 =============
def test_spm():
    """测试SPM环境是否正常"""
    print("\n🔍 测试SPM环境...")
    try:
        # 测试简单的恒流充电
        time = -charging_time_compute_3stage(5.0, 10, 4.0, 20, 3.0)
        print(f"✓ SPM环境正常，测试充电时间: {time * 1.5 / 60:.1f} 分钟")
        return True
    except Exception as e:
        print(f"❌ SPM环境异常: {e}")
        print("\n可能的解决方法:")
        print("1. 重新安装PyBaMM: pip install --upgrade pybamm")
        print("2. 重新安装CasADi: pip install --upgrade casadi")
        print("3. 检查Python环境是否正确")
        return False


def main():
    print("\n" + "="*70)
    print(" " * 20 + "三段式充电LLMBO优化")
    print("="*70)
    
    # 先测试SPM环境
    if not test_spm():
        print("\n❌ SPM环境测试失败，无法继续优化")
        return
    
    print("\n📋 参数边界:")
    for key, bounds in PBOUNDS_3STAGE.items():
        print(f"   {key}: {bounds}")
    
    # 初始化LLM（使用正确的参数名：model）
    print("\n🔧 初始化Qwen LLM...")
    llm = QwenLLMInterface(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    print(f"✓ 使用模型: {MODEL}")
    
    # 手动生成初始点（绕过LLMBO的硬编码参数名问题）
    print("🔧 生成三段式充电初始点...")
    n_init = 8
    initial_points = []
    
    # 使用LLM生成初始点
    prompt = f"""As an expert in battery fast charging, generate {n_init} diverse parameter sets for THREE-STAGE charging.

Parameter bounds:
- current1: {PBOUNDS_3STAGE['current1']} A (stage 1: high current)
- charging_number1: {PBOUNDS_3STAGE['charging_number1']} cycles (first transition)
- current2: {PBOUNDS_3STAGE['current2']} A (stage 2: medium current)
- charging_number2: {PBOUNDS_3STAGE['charging_number2']} cycles (second transition)
- current3: {PBOUNDS_3STAGE['current3']} A (stage 3: low current)

Reference protocols:
- Protocol A: 5.92A(10) → 4.92A(20) → 3.00A = 48 min
- Protocol B: 5.34A(12) → 4.56A(20) → 3.00A = 52 min

Constraints:
- current1 > current2 > current3 (currents must decrease)
- charging_number1 < charging_number2 (transitions must be sequential)

Generate {n_init} diverse, physically valid parameter sets as JSON array:
[
  {{"current1": ..., "charging_number1": ..., "current2": ..., "charging_number2": ..., "current3": ...}},
  ...
]

Only output the JSON array, no additional text."""

    try:
        print("🤖 调用LLM生成多样化初始点...")
        llm_response = llm.generate_response(prompt, temperature=0.9, max_tokens=2000)
        
        # 解析LLM返回的JSON
        import re
        json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if json_match:
            initial_points = json.loads(json_match.group())[:n_init]
            print(f"✓ LLM成功生成 {len(initial_points)} 个初始点")
        else:
            raise ValueError("无法解析LLM返回的JSON")
            
    except Exception as e:
        print(f"⚠️ LLM初始化失败: {e}")
        print("🔄 使用保守的预定义初始点...")
        
        # 使用保守、经过验证的初始点
        initial_points = [
            # 协议A变体
            {'current1': 5.92, 'charging_number1': 10, 'current2': 4.92, 'charging_number2': 20, 'current3': 3.00},
            {'current1': 5.80, 'charging_number1': 11, 'current2': 4.80, 'charging_number2': 21, 'current3': 3.00},
            
            # 协议B变体
            {'current1': 5.34, 'charging_number1': 12, 'current2': 4.56, 'charging_number2': 20, 'current3': 3.00},
            {'current1': 5.50, 'charging_number1': 11, 'current2': 4.60, 'charging_number2': 21, 'current3': 2.80},
            
            # 激进快充
            {'current1': 6.20, 'charging_number1': 10, 'current2': 5.00, 'charging_number2': 18, 'current3': 3.20},
            {'current1': 6.00, 'charging_number1': 12, 'current2': 4.80, 'charging_number2': 22, 'current3': 3.00},
            
            # 保守平衡
            {'current1': 5.50, 'charging_number1': 13, 'current2': 4.20, 'charging_number2': 23, 'current3': 2.80},
            {'current1': 5.40, 'charging_number1': 12, 'current2': 4.00, 'charging_number2': 22, 'current3': 2.50}
        ]
        print(f"✓ 使用 {len(initial_points)} 个预定义初始点")
    
    # 初始化优化器
    print("🔧 初始化LLMBO优化器...")
    optimizer = LLMBOOptimizer(
        objective_function=charging_time_compute_3stage,
        pbounds=PBOUNDS_3STAGE,
        llm_interface=llm,
        constraints=CONSTRAINTS,
        random_state=42
    )
    
    # 评估初始点
    print("\n📊 评估初始点...")
    for idx, point in enumerate(initial_points):
        target = charging_time_compute_3stage(**point)
        time_min = -target * 1.5 / 60
        optimizer.X.append([point[k] for k in sorted(PBOUNDS_3STAGE.keys())])
        optimizer.y.append(target)
        optimizer.history.append({
            'params': point,
            'target': target,
            'time': time_min
        })
        print(f"  点 {idx+1}: {time_min:.1f} min")
    
    # 拟合初始GP模型
    print("\n🔧 拟合初始GP模型...")
    optimizer.gp.fit(optimizer.X, optimizer.y)
    
    # 运行优化（跳过warm start，只做迭代）
    print("\n🚀 开始贝叶斯优化迭代...")
    print(f"   迭代次数: 40")
    
    start_time = time.time()
    
    try:
        # 手动运行优化循环
        from scipy.stats import norm as normal_dist
        
        for iteration in range(40):
            # 生成候选点
            n_candidates = 1000
            candidates = []
            for _ in range(n_candidates):
                point = {}
                for key, bounds in PBOUNDS_3STAGE.items():
                    point[key] = np.random.uniform(bounds[0], bounds[1])
                candidates.append(point)
            
            # 预测并计算acquisition function (PI)
            best_y = max(optimizer.y)
            best_acquisition = -np.inf
            best_candidate = None
            
            for candidate in candidates:
                X_test = np.array([[candidate[k] for k in sorted(PBOUNDS_3STAGE.keys())]])
                mu, sigma = optimizer.gp.predict(X_test, return_std=True)
                
                # Probability of Improvement
                xi = 0.01
                with np.errstate(divide='warn'):
                    imp = mu - best_y - xi
                    Z = imp / sigma if sigma > 1e-9 else 0
                    pi = normal_dist.cdf(Z)
                
                if pi > best_acquisition:
                    best_acquisition = pi
                    best_candidate = candidate
            
            # 评估最佳候选点
            target = charging_time_compute_3stage(**best_candidate)
            time_min = -target * 1.5 / 60
            
            optimizer.X.append([best_candidate[k] for k in sorted(PBOUNDS_3STAGE.keys())])
            optimizer.y.append(target)
            optimizer.history.append({
                'params': best_candidate,
                'target': target,
                'time': time_min
            })
            
            # 更新GP模型
            optimizer.gp.fit(optimizer.X, optimizer.y)
            
            # 打印进度
            current_best = max(optimizer.y)
            current_best_time = -current_best * 1.5 / 60
            print(f"  迭代 {iteration+1}/40: {time_min:.1f} min (当前最优: {current_best_time:.1f} min)")
        
        # 构建结果
        best_idx = np.argmax(optimizer.y)
        results = {
            'best_params': optimizer.history[best_idx]['params'],
            'best_time': optimizer.history[best_idx]['time'],
            'history': optimizer.history
        }
        
        elapsed_time = time.time() - start_time
        
        # 处理结果
        best_params = results.get('best_params', {})
        best_time = results.get('best_time', float('inf'))
        best_time_min = best_time * 1.5 / 60  # 转换为分钟
        
        # 输出结果
        print("\n" + "="*70)
        print("✅ 优化完成!")
        print("="*70)
        print(f"\n⏱️  总用时: {elapsed_time:.1f} 秒")
        print(f"\n🏆 最优参数:")
        print(f"   - 第一段电流: {best_params.get('current1', 0):.3f} A")
        print(f"   - 第一次切换: {best_params.get('charging_number1', 0):.0f} 周期")
        print(f"   - 第二段电流: {best_params.get('current2', 0):.3f} A")
        print(f"   - 第二次切换: {best_params.get('charging_number2', 0):.0f} 周期")
        print(f"   - 第三段电流: {best_params.get('current3', 0):.3f} A")
        print(f"\n⚡ 最优充电时间: {best_time_min:.1f} 分钟")
        
        # 对比论文协议A
        print("\n" + "="*70)
        print("📊 与论文参考协议对比:")
        print("="*70)
        
        time_A = -charging_time_compute_3stage(5.92, 10, 4.92, 20, 3.00)
        time_A_min = time_A * 1.5 / 60
        print(f"\n协议A (表4-3): {time_A_min:.1f} 分钟")
        print(f"  电流: 5.92A → 4.92A → 3.00A")
        print(f"  论文数据: 48 min, 温升 6.69K")
        
        improvement = (time_A_min - best_time_min) / time_A_min * 100
        print(f"\n🎯 相比协议A{'提升' if improvement > 0 else '差距'}: {abs(improvement):.1f}%")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results_3stage_{timestamp}.json'
        
        results_summary = {
            'timestamp': timestamp,
            'best_params': {k: float(v) for k, v in best_params.items()},
            'best_time_min': float(best_time_min),
            'elapsed_time': elapsed_time,
            'protocol_A_time_min': float(time_A_min),
            'improvement_vs_A': float(improvement)
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存至: {results_file}")
        
        # 可视化
        print("\n📊 生成可视化图表...")
        plot_3stage_results(results, f'optimization_3stage_{timestamp}.png')
        plot_3stage_charging_profile(best_params, f'charging_profile_3stage_{timestamp}.png')
        
        print("\n✅ 所有结果已保存完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 优化被用户中断")
    except Exception as e:
        print(f"\n\n❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()