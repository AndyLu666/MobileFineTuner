#!/usr/bin/env python3
"""
LoRA微調日誌查看工具
用於分析和可視化GPT2和Qwen LoRA微調的訓練日誌
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import seaborn as sns

def read_training_log(log_file):
    """讀取訓練日誌文件"""
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '[INFO]' in line and ('Loss=' in line or '完成' in line):
                logs.append(line.strip())
    return logs

def read_metrics_csv(csv_file):
    """讀取指標CSV文件"""
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"讀取CSV文件錯誤: {e}")
        return None

def plot_training_curves(df, output_dir="plots"):
    """繪製訓練曲線"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 過濾掉EPOCH_END行
    train_df = df[df['step'] != 'EPOCH_END'].copy()
    train_df['step'] = train_df['step'].astype(int)
    
    # 1. Loss曲線
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for epoch in train_df['epoch'].unique():
        epoch_data = train_df[train_df['epoch'] == epoch]
        plt.plot(epoch_data['step'], epoch_data['loss'], 
                label=f'Epoch {epoch}', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('每步Loss變化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 平均Loss曲線
    plt.subplot(2, 2, 2)
    for epoch in train_df['epoch'].unique():
        epoch_data = train_df[train_df['epoch'] == epoch]
        plt.plot(epoch_data['step'], epoch_data['avg_loss'], 
                label=f'Epoch {epoch}', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('平均Loss變化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 步驟時間分佈
    plt.subplot(2, 2, 3)
    plt.hist(train_df['step_time_ms'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Step Time (ms)')
    plt.ylabel('Frequency')
    plt.title('步驟時間分佈')
    plt.grid(True, alpha=0.3)
    
    # 4. Loss vs 時間
    plt.subplot(2, 2, 4)
    plt.plot(train_df['timestamp'], train_df['loss'], alpha=0.7)
    plt.xlabel('時間')
    plt.ylabel('Loss')
    plt.title('Loss隨時間變化')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_metrics(df):
    """分析訓練指標"""
    train_df = df[df['step'] != 'EPOCH_END'].copy()
    train_df['step'] = train_df['step'].astype(int)
    
    print("=== 訓練指標分析 ===")
    print(f"總訓練步數: {len(train_df)}")
    print(f"訓練epochs: {train_df['epoch'].nunique()}")
    print(f"最終Loss: {train_df['loss'].iloc[-1]:.6f}")
    print(f"最低Loss: {train_df['loss'].min():.6f}")
    print(f"平均步驟時間: {train_df['step_time_ms'].mean():.1f}ms")
    print(f"最慢步驟時間: {train_df['step_time_ms'].max():.1f}ms")
    
    # 每個epoch的統計
    print("\n=== 每個Epoch統計 ===")
    for epoch in sorted(train_df['epoch'].unique()):
        epoch_data = train_df[train_df['epoch'] == epoch]
        print(f"Epoch {epoch}:")
        print(f"  - 步數: {len(epoch_data)}")
        print(f"  - 平均Loss: {epoch_data['loss'].mean():.6f}")
        print(f"  - 最終Loss: {epoch_data['loss'].iloc[-1]:.6f}")
        print(f"  - 平均步驟時間: {epoch_data['step_time_ms'].mean():.1f}ms")

def main():
    parser = argparse.ArgumentParser(description='LoRA微調日誌查看工具')
    parser.add_argument('--log_dir', type=str, required=True, help='日誌目錄路徑')
    parser.add_argument('--plot', action='store_true', help='生成圖表')
    parser.add_argument('--analyze', action='store_true', help='分析指標')
    
    args = parser.parse_args()
    
    # 查找metrics.csv文件
    csv_files = []
    for file in os.listdir(args.log_dir):
        if file.endswith('_metrics.csv'):
            csv_files.append(os.path.join(args.log_dir, file))
    
    if not csv_files:
        print(f"在{args.log_dir}中未找到metrics CSV文件")
        return
    
    for csv_file in csv_files:
        print(f"\n處理文件: {csv_file}")
        df = read_metrics_csv(csv_file)
        
        if df is not None:
            if args.analyze:
                analyze_metrics(df)
            
            if args.plot:
                model_name = os.path.basename(csv_file).replace('_metrics.csv', '')
                output_dir = os.path.join(args.log_dir, f"{model_name}_plots")
                plot_training_curves(df, output_dir)
                print(f"圖表已保存到: {output_dir}")

if __name__ == "__main__":
    main()
