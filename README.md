# cube-analyzer

群論に基づくルービックキューブの表現・可視化と、それを「計測器」として使った
解法アプローチの研究リポジトリ。古典ソルバー、深層学習、LLMエージェントを
同じキューブ表現の上で並行して試している。

最終的には Swift 製の3Dアプリ（SceneKit / RealityKit + MLX Swift 埋め込み）への
搭載を目指す。

## トラック一覧

| # | トラック | 中身 | 状態 |
|---|---|---|---|
| 1 | [群論的表現・可視化](#1-群論的表現可視化) | `State` クラス／展開図 | 完成・テスト済み |
| 2 | [古典ソルバー (CFOP)](#2-古典ソルバー-cfop) | Cross→F2L→OLL→PLL | 完成 |
| 3 | [深層学習による解法](#3-深層学習による解法) | Transformer(BC) / 拡散(D3PM) / 価値反復 | 実験的（記録済み） |
| 4 | [LLM擬似直感エージェント](#4-llm擬似直感エージェント) | ツール化＋直感の付与 | 実験完了・記事化 |
| 5 | [Swift 3Dアプリ](#5-swift-3dアプリ) | インタラクティブ可視化＋MLX | 計画中 |

---

### 1. 群論的表現・可視化

状態と操作を同一の `State` クラスで表現する。状態遷移を群演算として扱い、
合成・逆元・単位元・繰り返しをすべて `State` オブジェクトで統一する
（`source/cube/state.py`）。

可視化（`source/cube/vis_util.py`）は、状態を 6×3×3 の展開図テンソルに変換し、
ASCII でアンフォールド表示する。独立した3Dフェイスレット・オラクル
（`tests/oracle.py`）で全手・ランダムスクランブルを検証している。

#### 群演算 API

| 演算 | 意味 |
|---|---|
| `a @ b` | `a` に `b` を適用（群の合成） |
| `a @= b` | `a` をインプレースで更新 |
| `~a` | 逆元 |
| `n * a` | 操作 `a` を `n` 回繰り返し（負数は逆方向） |
| `a == b` | 状態の等価比較（twist/flip の整数表現で） |
| `State()` | 解いた状態（単位元） |

基本操作: `MOVES['U']`, `'D'`, `'L'`, `'R'`, `'F'`, `'B'`

### 2. 古典ソルバー (CFOP)

`source/cfop.py`。Cross→F2L→OLL→PLL の4段。Cross/F2L は BFS、OLL/PLL は
最終層（62,208要素の部分群）上で事前構築したテーブルで解く。主に深層学習用の
教師データ（behavioral cloning）生成に使う。

### 3. 深層学習による解法

MLX（Apple Silicon）で2系統を試した。

1. **Transformer (BC)** — 状態＋操作履歴をトークン列とし、次手を自己回帰予測
2. **離散拡散 (D3PM)** — ランダム操作の適用を前向き拡散とみなし逆過程を学習

主な所見（詳細は [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)）:
ロールアウト成功率では拡散が BC を上回るが頭打ち。goal条件付けが学習されて
いない構造的問題があり、価値反復（DeepCubeA/DAVI）をヒューリスティックに
使う試みは退行した（負の結果として記録）。現在このトラックはパーク中。

### 4. LLM擬似直感エージェント

キューブ操作系を関数ツール化し、ローカルの小型LLM（Ollama 上の qwen3.5:4b/9b）に
解かせた実験。「小型LLMが難問で詰まるとき足りないのはツールか、先読みか、直感か」を、
条件を一つずつ変える ablation で切り分ける。

- `source/cube_tools.py` — 観測・プレビュー・確定・巻き戻し・直感・記憶のツールkernel
- `source/llm_agent.py` — Ollama駆動のエージェントハーネス
- `source/ablation_baselines.py` — 決定論ベースライン（床／天井）

主な結論: ツールだけでは解けない → 先読みだけでも足りない → **直感（学習済み価値
関数で「解への近さ」を渡すこと）が決定的**。さらに「直感をどれだけ信じるか」が
性能を律速し、モデルを大きくすると改善する。自己改善（手順の記憶）は4bでは起きなかった。

記事: [`docs/article_llm_intuition.md`](docs/article_llm_intuition.md)

### 5. Swift 3Dアプリ

SceneKit / RealityKit によるインタラクティブ3D表示と、MLX Swift モデルの
埋め込みを計画中（`swift/`）。

---

## 環境

Python 3.12 / macOS (Apple Silicon)。依存は [`uv`](https://docs.astral.sh/uv/) で管理する
（runtime: `mlx`、dev: `pytest`, `pytest-xdist`）。LLMトラックはローカルの Ollama を使う。

```bash
uv sync                 # .venv 作成＋依存インストール
uv run pytest           # テスト全実行
uv run pytest -n auto    # 並列実行
```

各モジュールには手動確認用の `__main__` ブロックがある。

```bash
cd source/cube
uv run python state.py      # 全手の identity / inverse / order-4 チェック
uv run python vis_util.py   # solved＋6つの基本手の展開図を表示
```

LLMエージェントの再現例（Ollama が必要）:

```bash
uv run python source/ablation_baselines.py --n 200 --depths 1-12
uv run python source/llm_agent.py --model qwen3.5:4b --mode intuition --depth 5 --episodes 20
```

## 開発上の注意

- 状態を変更する前に `.clone()` を使う（`apply` はミュータブル、`get_applied` / `@` はイミュータブル）
- GitHub Actions は使わない（セルフホストランナーを予定）
- フレームワークは MLX に統一（旧 PyTorch/MPS コードは移行済み）
