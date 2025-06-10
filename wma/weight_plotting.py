import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"  # Opens the plot in a web browser
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import argparse

parser = argparse.ArgumentParser(description="Compare voting strategies and plot model weights.")
parser.add_argument("--wma_dir", type=str, required=True, help="Directory containing WMA results.")
parser.add_argument("--rwma_dir", type=str, required=True, help="Directory containing RWMA results.")
# parser.add_argument("--naive_dir", type=str, required=True, help="Directory containing Naive results.")
parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset used.")
parser.add_argument("--data_type", type=str, required=True, help="Type of data (Train/Test).")
parser.add_argument("--output_dir", type=str, default="data/pic", help="Output directory for plots.")
parser.add_argument("--figure_number", type=str, required=True, help="Figure number for file naming.")
args = parser.parse_args()

# === 1) 配置路径与参数 ===
strategy_dirs = {
    'WMA': args.wma_dir,
    'RWMA': args.rwma_dir,
    # 'Naive': args.naive_dir
}
dataset_type = args.dataset_type
data_type = args.data_type
output_dir = args.output_dir
figure_number = args.figure_number

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 模型重命名字典
rename_dict = {
    "gpt-4.1-2025-04-14": "GPT 4.1 (2025-04-14)",
    "grok3": "Grok 3 beta",

}

# === 2) 讀取所有策略的數據 ===
all_data = {}
for strategy, result_dir in strategy_dirs.items():
    file_path = os.path.join(result_dir, "results_1.json")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        for entry in data:
            rnd = entry["index"]
            weights = entry["current_weights"]
            for model, w in weights.items():
                records.append({
                    "Round": rnd, 
                    "Model": model, 
                    "Weight": w, 
                    "Strategy": strategy
                })
        
        all_data[strategy] = pd.DataFrame(records)
        print(f"Loaded {strategy}: {len(all_data[strategy])} records from {file_path}")
    else:
        print(f"Warning: File not found for strategy {strategy}: {file_path}")

# 合併所有策略的數據
if not all_data:
    raise ValueError("No valid strategy data found!")

df_combined = pd.concat(all_data.values(), ignore_index=True)

# 重命名模型
df_combined['Model'] = df_combined['Model'].map(rename_dict).fillna(df_combined['Model'])

# === 3) Plotly 互動式圖表 ===
fig = px.line(
    df_combined,
    x="Round",
    y="Weight",
    color="Model",
    line_dash="Strategy",
    markers=True,
    hover_data={"Model": True, "Weight": ":.4f", "Strategy": True},
    template="plotly_white",
    title=f"Voting Strategy Comparison on {dataset_type} {data_type}"
)

# 更新圖表樣式
fig.update_layout(
    title={
        "text": f"Voting Strategy Comparison on {dataset_type} {data_type}",
        "x": 0.5, "xanchor": "center", "yanchor": "top",
        "font": {"size": 20, "family": "Arial", "color": "black"}
    },
    annotations=[{
        "text": f"Comparing {', '.join(strategy_dirs.keys())} Strategies",
        "x": 0, "xref": "paper",
        "y": 1.02, "yref": "paper",
        "showarrow": False,
        "align": "left",
        "font": {"size": 14, "family": "Arial", "color": "gray"}
    }],
    legend={
        "title": "",
        "orientation": "v",
        "x": 1.02, "xanchor": "left", "y": 1,
        "font": {"size": 10},
        "traceorder": "normal"
    },
    margin={"l": 60, "r": 250, "t": 140, "b": 60}
)

fig.update_xaxes(title_text="Rounds", showgrid=True, gridwidth=1, gridcolor="#EEEEEE")
fig.update_yaxes(title_text="Model Weight", showgrid=True, gridwidth=1, gridcolor="#EEEEEE")

fig.show()

# === 4) Matplotlib 論文風格圖表 - 子圖比較 ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
})

# 為每個策略創建子圖
strategies = list(strategy_dirs.keys())
n_strategies = len(strategies)
fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5), dpi=300)
if n_strategies == 1:
    axes = [axes]

# 線條樣式和標記
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', 'h']

for idx, strategy in enumerate(strategies):
    ax = axes[idx]
    
    # 獲取該策略的數據
    strategy_data = df_combined[df_combined['Strategy'] == strategy]
    df_pivot = strategy_data.pivot(index="Round", columns="Model", values="Weight").fillna(0)
    
    if df_pivot.empty:
        print(f"Warning: No data found for strategy {strategy}")
        continue
        
    rounds = df_pivot.index.values
    models = df_pivot.columns.tolist()
    
    # 繪製每個模型的折線
    for i, model in enumerate(models):
        style = linestyles[i % len(linestyles)]
        m = markers[i % len(markers)]
        ax.plot(rounds, df_pivot[model],
                linestyle=style,
                marker=m,
                markevery=max(1, len(rounds)//20),
                markersize=4,
                linewidth=1.2,
                label=model)
    
    # 設置坐標軸
    ax.set_xlabel('Rounds', labelpad=6)
    if idx == 0:  # 只在第一個子圖顯示 y 軸標籤
        ax.set_ylabel('Model Weight', labelpad=6)
    
    ax.set_title(f'{strategy}', fontsize=14, fontweight='bold', pad=10)
    
    # 網格和刻度
    if len(rounds) > 1:
        ax.set_xticks(np.linspace(rounds.min(), rounds.max(), min(6, len(rounds)), dtype=int))
    ax.set_yticks(np.linspace(0, df_pivot.values.max(), 5))
    ax.grid(which='major', linestyle='-', linewidth=0.6, color='grey', alpha=0.4)
    ax.grid(which='minor', linestyle=':', linewidth=0.4, color='grey', alpha=0.2)
    ax.minorticks_on()
    
    # 去掉上、右框線
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 總標題
fig.suptitle(
    f"Voting Strategy Model Weights Comparison on {dataset_type} {data_type}",
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# 統一圖例（放在右側）
if len(axes) > 0:
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # 確保有圖例內容
        fig.legend(handles, labels,
                   loc='center left',
                   bbox_to_anchor=(1.0, 0.5),
                   fontsize=9,
                   handlelength=1.8,
                   labelspacing=0.3)

plt.tight_layout()
plt.subplots_adjust(right=0.85)

# 保存子圖比較
subplot_filename = os.path.join(output_dir, f'model_weights_comparison_{figure_number}.png')
plt.savefig(subplot_filename, dpi=300, bbox_inches='tight')
print(f"Saved subplot comparison: {subplot_filename}")
plt.show()

# === 5) Matplotlib 單一圖表顯示所有策略 ===
fig_single, ax_single = plt.subplots(figsize=(12, 8), dpi=300)

strategy_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
strategy_linestyles = ['-', '--', '-.']

for s_idx, strategy in enumerate(strategies):
    strategy_data = df_combined[df_combined['Strategy'] == strategy]
    df_pivot = strategy_data.pivot(index="Round", columns="Model", values="Weight").fillna(0)
    
    if df_pivot.empty:
        continue
        
    rounds = df_pivot.index.values
    models = df_pivot.columns.tolist()
    
    base_color = strategy_colors[s_idx % len(strategy_colors)]
    base_linestyle = strategy_linestyles[s_idx % len(strategy_linestyles)]
    
    for i, model in enumerate(models):
        alpha = 0.7 + 0.3 * (i / max(1, len(models)-1))  # 不同模型用不同透明度
        ax_single.plot(rounds, df_pivot[model],
                      linestyle=base_linestyle,
                      color=base_color,
                      alpha=alpha,
                      linewidth=1.5,
                      label=f'{model} ({strategy})')

ax_single.set_xlabel('Rounds', labelpad=6)
ax_single.set_ylabel('Model Weight', labelpad=6)
ax_single.set_title(
    f"All Strategies Comparison on {dataset_type} {data_type}",
    fontsize=16,
    fontweight='bold',
    pad=15
)

ax_single.grid(which='major', linestyle='-', linewidth=0.6, color='grey', alpha=0.4)
ax_single.spines['top'].set_visible(False)
ax_single.spines['right'].set_visible(False)

# 圖例放在右側，分列顯示
ax_single.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()

# # 保存單一圖表
# single_filename = os.path.join(output_dir, f'model_weights_all_strategies_{figure_number}.png')
# plt.savefig(single_filename, dpi=300, bbox_inches='tight')
# print(f"Saved single chart: {single_filename}")
# plt.show()

print(f"\nAll charts saved to: {output_dir}")
print(f"Files created:")
print(f"- model_weights_comparison_{figure_number}.png")
print(f"- model_weights_all_strategies_{figure_number}.png")


"""
python wma/weight_plotting.py \
--wma_dir spider2-lite/evaluation_suite/vote/log/gpt_4.1_grok3_vote_100_wma \
--rwma_dir spider2-lite/evaluation_suite/vote/log/gpt_4.1_grok3_vote_100_rwma \
--dataset_type "Spider 2.0-Lite" \
--data_type "The First 100 Queries" \
--output_dir "spider2-lite/evaluation_suite/vote/pic" \
--figure_number "2"

"""