# TensorAlpha 开源版

TensorAlpha 是一个面向股票截面排序的 Transformer 量化研究工具包，覆盖数据读取、无未来特征、模型训练、Purged K-Fold OOF、A 股规则事件回测和离线模拟盘。

公开仓库只包含源码、必要脚本、测试、文档、OOF 聚合统计、脱敏的 OOS 净值检查点和合成演示数据，不包含任何真实 key、邮箱、本机路径、原始行情、模型权重、证券级预测、持仓、订单或成交记录。

> 本项目只用于研究和工程学习，不构成投资建议。

## 可视化概览

下图来自 2015–2026 年共 10,725,368 条真实 Transformer OOF 预测的脱敏聚合统计。图中不包含证券级记录、价格、收益率或组合绩效。

- 柱形是当年生成的 OOF **预测条数**，不是股票数量。2015 年只覆盖 4 月 7 日至年末（184 个交易日），所以第一根柱子代表这段不完整年度内的 415,586 条预测；2026 年同样只覆盖到 4 月 10 日。
- “每日排名分数”是模型在同一天所有候选标的中的百分位排名，范围为 0–1。比如 0.90 表示该预测排在当天约 90% 的候选预测之前；它不是积分、价格、上涨概率或收益率。
- 深色曲线是每年的排名中位数；深色带包含中间 50% 的预测（P25–P75），浅色带包含中间 90%（P05–P95）。

![TensorAlpha OOF 年度覆盖与分位数曲线](docs/assets/oof_profile.svg)

第二张图展示 **TensorAlpha STK-O** 的脱敏研究结果。它是截至 2026-04-18 的三阶段综合比较冠军，每日排名由 ME 0.4、CE_Liq 0.2 与 V46ME S43 0.4 加权组成。之所以不用单独的 V46ME 种子，是因为综合比较同时看长期 OOS 与近期 OOS 的一致性，不按最高历史收益挑 seed。

标准研究口径为：Top 10、流通市值不低于 30 亿元、日成交额不低于 5000 万元、每个行业最多 2 只、T+1 执行，并计入 0.2% 滑点、0.03% 佣金和 0.05% 印花税。2015-04-07 至 2026-03-16 的 Purged K-Fold OOS 共 2,659 个交易日，累计收益 **+324.37%**、年化 **+14.68%**、最大回撤 **-55.88%**、Sharpe **0.5894**。其中 -55.88% 是非常深的回撤，图和文字都不回避它。

![TensorAlpha STK-O 十一年 OOS 净值曲线](docs/assets/stko_11y_oos.svg)

公开曲线只有日志中 55 个真实净值检查点：第 1 个交易日、此后每 50 个交易日以及最后一天；没有插值冒充日频数据。最大回撤来自原始完整日频回测，不是用稀疏检查点重新计算的。公开文件不含证券代码、价格、个股收益、订单、持仓、模型路径或账户信息。这是历史研究结果，不是实盘业绩，也不保证未来表现。

运行 `python scripts/render_showcase.py` 可以确定性地重新生成 README 中的两张 SVG；加上 `--include-synthetic` 可同时重建可执行流程的合成演示图，不需要额外绘图库。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
tensoralpha demo --output artifacts/demo --days 260 --assets 40
```

演示完全离线运行，证券代码均为 `DEMO0001` 形式，行情与收益全部为合成值；输出目录同时包含 `backtest_nav.svg`。

## 完整流程

```bash
# 可选：从 Tushare 获取合法授权的数据
pip install -e ".[data]"
$env:TENSORALPHA_TUSHARE_TOKEN="<your-token>"
tensoralpha fetch-data --start 2015-01-01 --end 2025-12-31 --output data/market.parquet

# 单模型训练
tensoralpha train --market data/market.parquet --output artifacts/models/transformer

# Purged K-Fold OOF
tensoralpha oof --market data/market.parquet --output artifacts/oof --folds 5 --purge-days 20

# 事件驱动回测
tensoralpha backtest --market data/market.parquet --signals artifacts/oof/oof_predictions.parquet --output artifacts/backtest

# 离线模拟盘
tensoralpha paper-create --account artifacts/paper/demo --initial-cash 1000000 --top-n 10
tensoralpha paper-tick --account artifacts/paper/demo --market one_day.csv --signals one_day_signals.csv
```

回测和模拟盘共享同一套 Top-N 策略与撮合器：T 日收盘后形成目标，T+1 开盘执行，并处理整手、停牌、涨跌停、佣金、印花税和滑点。

完整模块说明请从英文主 [README](README.md) 的文档地图进入。
