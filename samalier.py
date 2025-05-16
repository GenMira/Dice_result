import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

# スクリプトのあるディレクトリ
script_dir = Path(__file__).resolve().parent

# 対象の文字種
Sigma = "abcdefg"

# 全ての .txt ファイルを取得（report.txt は除外）
txt_files = [f for f in script_dir.glob("*.txt") if not f.name.endswith("_report.txt")]

for file_path in txt_files:
    # ファイル内容を読み取り
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if text[:5] == "Range":
      text =text[text.find(";")+1:]
    else:
      text = text[:text.find(";")]
    N = len(text)
    observed = [text.count(s) for s in Sigma]
    expected_probabilities = [1 / len(Sigma)] * len(Sigma)
    expected = [N * p for p in expected_probabilities]
    chi_squared = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    p_value = 1 - stats.chi2.cdf(chi_squared, df=len(Sigma) - 1)

    # 結果を文字列としてまとめる
    result_lines = [
        f"ファイル名: {file_path.name}",
        f"サンプル数: {N}",
    ]
    for s in Sigma:
        count = text.count(s)
        ratio = count / N * 100 if N > 0 else 0
        ideal = round(1 / len(Sigma) * 100, 2)
        result_lines.append(f"{s}面の実測値： {count} 回 ({ratio:.2f}%)  理想値: {ideal}%")

    result_lines.append(f"\nカイ二乗統計量: {chi_squared:.4f}")
    result_lines.append(f"これ以上の誤差になる確率: {p_value * 100:.4f} %")

    # 結果をレポートファイルに保存
    report_path = script_dir / f"{file_path.stem}_report.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(result_lines))

    # 時系列グラフの準備
    array = [np.array([0] * len(Sigma))]
    for i in range(N):
        array.append(array[-1] + np.array([1 if s == text[i] else 0 for s in Sigma]))

    init_bias = 500
    for s in Sigma:
        plt.plot(
            range(init_bias, N + 1),
            [array[i][Sigma.index(s)] / i if i > 0 else 0 for i in range(N + 1)][init_bias:],
            label=f"count_{s}"
        )

    plt.title(f"出現頻度の推移: {file_path.name}")
    plt.xlabel("文字数")
    plt.ylabel("出現割合")
    plt.legend()
    plt.tight_layout()

    # グラフ画像として保存
    graph_path = script_dir / f"{file_path.stem}_graph.png"
    plt.savefig(graph_path)
    plt.clf()  # 次のファイルのためにグラフを初期化

print("全てのファイルに対して処理が完了しました。")
