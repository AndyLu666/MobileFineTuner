#!/usr/bin/env python3
"""
解析训练日志并绘制 Loss 曲线
用法: python plot_loss_curve.py <log_file> [output_image]
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path):
    """解析训练日志，提取 step 和 loss"""
    steps = []
    losses = []
    
    # 匹配模式：[Step X] Loss=Y 或 step X/... | loss Y
    pattern1 = r'\[Step (\d+)\] Loss=([\d.]+)'
    pattern2 = r'step (\d+)/\d+.*loss ([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern1, line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                continue
            
            match = re.search(pattern2, line, re.IGNORECASE)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    
    return steps, losses

def plot_loss_curve(steps, losses, output_path, title="Training Loss Curve"):
    """绘制 loss 曲线"""
    plt.figure(figsize=(12, 6))
    
    # 主曲线
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.8, label='Loss')
    
    # 添加移动平均线
    window = min(20, len(losses) // 5) if len(losses) > 20 else 5
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        plt.plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Moving Avg (w={window})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{title} (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Loss 曲线已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(steps)}")
    print(f"  初始 Loss: {losses[0]:.4f}")
    print(f"  最终 Loss: {losses[-1]:.4f}")
    print(f"  最低 Loss: {min(losses):.4f} (Step {steps[losses.index(min(losses))]})")
    print(f"  平均 Loss (最后20步): {np.mean(losses[-20:]):.4f}")

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_loss_curve.py <log_file> [output_image]")
        print("示例: python plot_loss_curve.py runs/gemma_1b_lora_short_300steps/train.log")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    # 默认输出路径
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        import os
        base_dir = os.path.dirname(log_path)
        output_path = os.path.join(base_dir, "loss_curve.png")
    
    # 解析日志
    steps, losses = parse_log(log_path)
    
    if not steps:
        print(f"❌ 未能从日志中解析出 loss 数据: {log_path}")
        sys.exit(1)
    
    print(f"📈 解析到 {len(steps)} 个数据点")
    
    # 从日志路径推断标题
    if 'gemma_1b' in log_path.lower():
        title = "Gemma 1B LoRA Training Loss"
    elif 'gpt2_medium' in log_path.lower():
        title = "GPT-2 Medium LoRA Training Loss"
    elif 'gemma' in log_path.lower():
        title = "Gemma LoRA Training Loss"
    elif 'gpt2' in log_path.lower():
        title = "GPT-2 LoRA Training Loss"
    else:
        title = "Training Loss Curve"
    
    # 绘制曲线
    plot_loss_curve(steps, losses, output_path, title)

if __name__ == "__main__":
    main()

