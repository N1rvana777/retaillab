# RetailLab 系统架构

## 2026-09-21 现状校正（阅读原架构前必读）

原图同时包含设计目标和历史实验版本，不作为生产系统完成证明。当前执行目标见 `plan.md` 顶部，详细证据见 `docs/CONTEXT_AUDIT_20260921.md`。

- 实际主线：`data/raw/销售` → 旧 `orchestrator` → `data/features/features_*.parquet`；`data/raw/进货` 与 `库存` → `data_pipeline/ingest_v2` 的采购、快照、映射、分类、库存/需求估计 → 合并旧特征生成 `features_v2_*.parquet` → `scripts/train_v2.py` → `models/v2_quantile_models.joblib` → `scripts/profit_v2.py` 场景报告。
- V2依赖旧特征；原数据没有重新经过一条已验收的全新管道。滚动/EMA跨SKU污染、目标统计跨训练/验证、库存漏出库流量、单位与成本口径冲突，均待修复。
- 当前模型维数不统一：旧 lgbm存档89、step2/step3为64、V2为94；Parquet总列数与模型输入列数不同。下方“64特征”仅描述历史版本。
- “需求还原”是带假设的潜在需求估计，不是实测真值；“库存锚点误差零”由快照强制重置产生。年度成本率报告不等于多期真实利润回测。
- `quant_engine` 有单期报童/约束/降级等组件，独立训练与收益脚本未完全经过服务层。API初始化与部分路由仍是TODO/占位，Docker入口尚未端到端验证。`hedging` 无可用策略实现。
- `tests` 当前77项通过，主要覆盖既有数学/防御与模拟输入管道，不能为V2经济结论背书。暂不扩建部署层，先完成研究验证链。

以下保留原架构说明，阅读时结合以上版本与实现边界。

## 一、系统概述

RetailLab 是一个**连锁超市智能补货系统**，核心功能是：根据历史销售数据预测未来需求，结合报童模型（Newsvendor Model）给出最优补货量。

```
原始数据 (.xls)     →    数据清洗     →    特征工程     →    ML 预测     →    报童模型     →    补货建议
(科脉 POS + 管家婆)      (标记异常)       (64特征)       (LightGBM)       (成本优化)       (订多少货)
                                                          ↑
                                                    需求还原模块
                                              (缺货天真实需求估算)
```

## 二、三层架构（严格解耦）

```
┌─────────────────────────────────────────────────────────────┐
│                     api_gateway/                            │
│        FastAPI REST API + JWT 认证 + 限流                    │
│        唯一职责：接收请求 → 调用 quant_engine → 返回结果        │
│        只能通过 quant_engine.service 访问引擎                  │
└───────────────────────┬─────────────────────────────────────┘
                        │ 调用 service.py (Facade)
┌───────────────────────▼─────────────────────────────────────┐
│                     quant_engine/                            │
│        核心引擎：预测 + 优化 + 决策                             │
│                                                             │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│   │ forecasting/ │  │ optimization/│  │  monitoring/  │     │
│   │ LightGBM    │  │ Newsvendor   │  │  漂移检测      │     │
│   │ 冷启动       │  │ 约束处理      │  │  2σ/3σ 告警
   │     │                                 gate keeper
│   │ 集成模型     │  │ 损失函数      │  └───────────────┘     │
│   └─────────────┘  └──────────────┘                         │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│   │  training/  │  │  telemetry/  │  │   fallback/   │     │
│   │ 训练器       │  │  决策日志     │  │  安全库存兜底   │     │
│   │ TimeSeriesSplit│ │  审计追踪    │  │  7天均值×1.2  │     │
│   └─────────────┘  └──────────────┘  └───────────────┘     │
└─────────────────────────────────────────────────────────────┘
                        ▲ 读取 Parquet 特征文件
┌───────────────────────┴─────────────────────────────────────┐
│                     data_pipeline/                           │
│        数据管道：原始数据 → ML-ready 特征                       │
│                                                             │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│   │   ingest/   │  │  cleaning/   │  │   feature_    │     │
│   │ 科脉读取器   │  │  去重扫描     │  │ engineering/  │     │
│   │ 管家婆读取器 │  │  宕机检测     │  │  时间特征     │     │
│   │ 天气/节假日  │  │  折扣异常     │  │  滞后特征     │     │
│   │ 批量加载器   │  │  数量异常     │  │  滚动统计     │     │
│   └─────────────┘  │  补全填充     │  │  趋势因子     │     │
│                     └──────────────┘  │  天气/节假日  │     │
│   ┌─────────────┐                     │  产品特征     │     │
│   │  storage/   │                     │  穿越防护     │     │
│   │ Parquet存储  │                     └───────────────┘     │
│   │ + 原始缓存   │                                          │
│   └─────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

**红线规则**：
- `data_pipeline/` 绝不导入 `quant_engine/`
- `quant_engine/` 绝不导入 `data_pipeline/` 或 `api_gateway/`
- `api_gateway/` 只通过 `quant_engine.service.py`（Facade 模式）访问引擎
- 模块间通信只通过 numpy arrays、dataclasses、Parquet 文件

## 三、数据流（端到端）

### 3.1 数据清洗 Pipeline

```
科脉 POS .xls (292个文件, 1600万行)
    │
    │  kemai_reader.py
    │  - 自动检测每个文件的 header 格式（有些在第1行，有些在第2行）
    │  - 标准化列名（不同门店列不一样）
    │  - barcode: float→int→str（避免科学记数法）
    │  - 标记 store_id（store_a / store_b / store_c）
    │  - 标记 store_city（city_a / city_b，用于天气匹配）
    │
    ▼
batch_loader.py — 总调度
    │
    ├── 原始数据缓存：首次读 .xls（~10分钟）后保存 raw_pos.parquet
    │   以后直接读缓存（~1秒），--force-reload 强制重读
    │
    ├── 加载管家婆商品信息（18,049 产品）
    │   └── barcode_base: 去字母后缀用于匹配
    │
    ├── 加载管家婆进货统计（进货均价 / 规格 = 单件成本）
    │   └── 标记 is_promotional_gift（纯赠品，成本=0）
    │
    ├── 加载天气数据（weather.csv, Open-Meteo API, 城市A+城市B）
    │
    ├── 生成节假日特征（chinese-calendar + lunardate, 2024-2050）
    │   包含 17 个节日：春节/元宵/龙抬头/端午/七夕/中秋/重阳/腊八/小年
    │   + 元旦/情人节/妇女节/劳动节/儿童节/国庆/圣诞 + 调休标记
    │
    ├── 预处理：discount_amount → discount_pct, 标记 is_return
    │
    ▼
orchestrator.py — Pipeline 执行器
    │
    ├── [1] remove_duplicate_scans — 5秒窗口内重复扫描去重
    ├── [2] detect_outages — 系统宕机时段标记
    ├── [3] detect_discount_anomalies — 异常折扣检测修正
    ├── [4] detect_quantity_outliers — 数量离群值（均值+5σ）只标记不修改
    │       原始值保留，让模型自己决定如何处理
    ├── [5] detect_decimal_errors — 小数点错误（10x/0.1x）修正
    ├── [6] impute_outage_periods — 宕机时段销量补全
    ├── [7] _aggregate_daily — 逐笔交易聚合为日度 SKU-Store 级别
    │       GROUP BY (date, store_id, sku_id)
    │       quantity_sold: SUM, total_amount: SUM
    │       1600万行 → 1500万行日度记录
    │
    ├── [8] 特征工程（详见下方特征清单）
    │
    ├── [9] LeakageGuard — 验证所有 lag/rolling 特征无未来信息泄漏
    │
    └── [10] save_features → data/features/features.parquet
            ├── features_train.parquet   (2024全年, 6.8M行)
            ├── features_oos.parquet     (2025全年, 7.1M行)
            └── features_holdout.parquet (保留)
```

### 3.2 需求还原模块 (demand_uncensor.py)

```
POS 日度销售数据（censored：缺货时真实需求不可观测）
    │
    ├── [1] detect_stockouts — 缺货天识别
    │       完全缺货：活跃SKU（sell_rate≥30%）当天销量=0
    │       部分缺货：当天销量 < 正常水平(median)的50%
    │       合计影响 488,491 SKU-天（6.9%）
    │
    ├── [2] uncensor_demand — 真实需求估算
    │       用非缺货天的同星期平均销量填充缺货天
    │       结果：估计的真实需求高于观测需求（具体数字涉及合作方经营数据，公开版本已删除）
    │
    └── [3] 模型收入 = min(模型进货量, 真实需求) × 单价
            模型因备货充足，能捕获实际被缺货浪费的需求
```

### 3.3 特征清单（训练后清洗至 64 个）

#### 时间特征 (temporal_features.py) — 11 个
| 特征 | 说明 |
|------|------|
| day_of_week | 星期几 (0-6) |
| day_of_month | 几号 (1-31)，模型自己学月内消费周期 |
| week_of_year | 第几周 (1-53) |
| month | 月份 (1-12) |
| quarter | 季度 (1-4) |
| is_weekend | 是否周末 (0/1) |
| dow_sin / dow_cos | 星期几圆周编码（保留备用，LightGBM 主要用整数编码） |
| month_sin / month_cos | 月份圆周编码（同上） |
| year | 年份 |

#### 滞后特征 (lag_features.py) — 清洗后保留 ~20 个
| 特征 | 说明 |
|------|------|
| sales_lag_1 ~ sales_lag_7 | 近 7 天逐日 lag（全部保留） |
| sales_lag_14, 21, 28, 35 | 有周期意义的 lag（每周一个锚点） |
| 其余 lag_8~lag_35 | 与相邻 lag 相关>0.95 的已裁剪 |
| ~~sales_lag_yoy~~ | 训练仅 1 年无法计算，已删除 |

#### 滚动统计 (lag_features.py) — 12 个
| 特征 | 说明 |
|------|------|
| rolling_7d_mean/std/min/max | 近 7 天滚动统计（shift(1) 防穿越） |
| rolling_14d_mean/std/min/max | 近 14 天滚动统计 |
| rolling_28d_mean/std/min/max | 近 28 天滚动统计 |

#### 趋势因子 (lag_features.py) — 5 个
从 crypto 量化策略迁移的因子：

| 特征 | 公式 | 说明 |
|------|------|------|
| momentum_7d | shift(1) - shift(8) | 7 天销量绝对变化 |
| momentum_7d_ratio | (shift(1) - shift(8)) / shift(8) | 7 天变化率 |
| bollinger_position | (shift(1) - rolling_7d_mean) / rolling_7d_std | 昨天销量的 z-score |
| ema_7 | 7 天指数移动平均 (α=0.25) | 短期趋势，对近期敏感 |
| ema_28 | 28 天指数移动平均 (α=0.07) | 中期趋势，两者比较=金叉/死叉 |

#### 异常事件特征 (anomaly_features.py) — 3 个
| 特征 | 说明 |
|------|------|
| is_demand_anomaly_low | 当天是否有低需求异常（宕机） |
| is_demand_anomaly_high | 当天是否有高需求异常（数量飙升） |
| days_since_last_anomaly | 距上次异常几天（截断 90，捕捉恢复曲线） |

#### 节假日特征 (holiday_calendar.py) — 6 个
使用 chinese-calendar（官方假日+调休）+ lunardate（农历节日），覆盖 2024-2050。

| 特征 | 说明 |
|------|------|
| is_holiday | 是否官方假日 |
| is_festival | 是否传统节日（农历+公历，17个） |
| holiday_name | 节日名称 |
| days_to_nearest_holiday | 距最近节假日天数（连续值，模型自己学备货提前量） |
| nearest_holiday_name | 最近是哪个节日（让模型区分春节 vs 清明的不同效应） |
| is_makeup_workday | 是否调休上班日（周末但要上班） |

包含的 17 个节日：
春节、元宵、龙抬头、端午、七夕、中秋、重阳、腊八、小年、
元旦、情人节、妇女节、劳动节、儿童节、国庆、圣诞 + 官方调休假日

#### 天气特征 (weather_client.py + external_features.py) — 15 个
Open-Meteo API 获取，按 (date, store_city) 匹配，城市A和城市B各有独立天气。

| 特征 | 说明 |
|------|------|
| temp_max / temp_min / temp_mean | 当天最高/最低/平均温度 |
| precipitation / humidity / wind_speed_max | 降水/湿度/最大风速 |
| temp_range | 温差 |
| is_rainy / is_extreme_heat / is_extreme_cold | 雨天/极热(>35°C)/极寒(<-10°C，东北标准) |
| weather_temp_mean_lag1 | 昨日平均温度 |
| weather_precipitation_lag1 | 昨日降水 |
| weather_humidity_lag1 | 昨日湿度 |
| weather_temp_3d_avg | 3日均温趋势 |
| store_city_cn | 城市中文名 |

#### 产品特征 (product_features.py) — 通过 barcode 匹配管家婆
| 特征 | 说明 |
|------|------|
| unit_cost | 单件进货成本（进货均价/规格，覆盖率 16%，待科脉编码映射表提升） |
| is_promotional_gift | 是否纯促销赠品（排除出补货模型） |

#### 训练时排除的列（存在但不喂给模型）
| 列 | 排除原因 |
|----|---------|
| total_amount | 当天销售额，预测时不可知，存在数据穿越 |
| sku_name / spec / barcode 等 | 字符串，非数值特征 |
| quantity_sold | 预测目标本身 |

### 3.4 模型训练

```
features_train.parquet (2024全年, 6.8M行)
    │
    │  scripts/train_model.py + scripts/quantile_compare.py
    │
    ▼
LightGBM 分位数回归 × 9 个分位数
    │  64 特征（清洗后）
    │  sample_weight = log1p(quantity_sold)（高频SKU优先）
    │
    │  q = 0.50, 0.55, 0.60, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95
    │  推荐方案：q=0.75（利润最优 sweet spot）
    │
    ▼
models/step3_results.joblib
    │
    ▼
InventoryOptimizer — 接入报童模型
    │
    │  1. 计算临界比率 cr = Cu/(Cu+Co)
    │  2. 在 cr 对应的分位数处预测需求 → 最优订货量
    │  3. 应用约束（MOQ/货架容量/整车）
    │  4. 记录决策日志（telemetry）
    │
    ▼
ReplenishmentOrder — 补货建议
    包含：推荐数量、预测均值、置信区间、预期成本、使用方法
```

### 3.5 利润对比方法

```
实际利润 = POS 收入 - 三店进货成本（具体数字涉及合作方经营数据，公开版本已删除）

模型利润计算：
  1. 日度预测 → 年度汇总 → 加5%年度缓冲（不逐日加，防累积膨胀）
  2. 模型收入 = min(模型进货量, 真实需求) × 单价
     ↑ 真实需求 = 观测销量 + 缺货还原量（demand uncensoring）
  3. 模型成本 = 模型收入 × COGS ratio（全店统一）
     ↑ 用实际成本收入比，避免SKU混合导致的平均单价偏差
  4. 模型利润 = 模型收入 - 模型成本

最终结果（q=0.75推荐方案）：
  模型利润 1.966亿 vs 实际 1.810亿 → +1569万（+8.7%）
```

### 3.6 防御体系（6 大防御模块）

```
┌─────────────────────────────────────────────────────────┐
│  1. telemetry/ — 决策日志                                │
│     每笔补货决策写入 JSONL，包含输入特征、预测值、           │
│     最终订量、使用方法、延迟时间，可审计可回溯               │
│                                                         │
│  2. training/evaluator.py — 误差分析                     │
│     Top 5% 最差预测分析，定位系统性偏差                     │
│     Pinball Loss + 成本加权评估                           │
│                                                         │
│  3. monitoring/drift_detector.py — 漂移监控               │
│     跟踪 Pinball Loss 的滑动均值                          │
│     超 2σ 告警，超 3σ 触发模型重训                         │
│                                                         │
│  4. feature_engineering/leakage_guard.py — 穿越防护       │
│     验证 lag/rolling 特征与目标的相关性 < 0.95              │
│     防止未来数据泄漏导致虚假精度                            │
│                                                         │
│  5. fallback/safety_stock.py — 优雅降级                   │
│     ML 引擎挂了 → 降级到 7天均值×1.2 安全库存               │
│     永远不返回空结果，永远有一个兜底值                       │
│                                                         │
│  6. service.py — 双层 try-except                         │
│     主引擎失败 → 安全库存兜底 → 极端情况返回 0 + 错误信息     │
│     对外保证：永不抛异常，永远返回有效结果                    │
└─────────────────────────────────────────────────────────┘
```

## 四、关键文件速查

| 文件 | 职责 | 重要程度 |
|------|------|---------|
| `data_pipeline/ingest/kemai_reader.py` | 读取科脉 POS 原始 .xls，多门店多格式 | ★★★ |
| `data_pipeline/ingest/batch_loader.py` | 一键跑完整 pipeline，含原始数据缓存 | ★★★ |
| `data_pipeline/orchestrator.py` | Pipeline 执行器，串联清洗+特征+存储 | ★★★ |
| `data_pipeline/cleaning/misrecord_handler.py` | 去重+数量异常(只标记不修改)+小数错误 | ★★ |
| `data_pipeline/feature_engineering/lag_features.py` | lag 1-35 + yoy + rolling + 趋势因子 + EMA | ★★★ |
| `data_pipeline/feature_engineering/leakage_guard.py` | 穿越防护验证器 | ★★ |
| `data_pipeline/ingest/holiday_calendar.py` | 节假日（chinese-calendar + lunardate, 17个节日, 2024-2050） | ★★ |
| `quant_engine/optimization/newsvendor.py` | 报童模型数学核心 | ★★★ |
| `quant_engine/optimization/inventory_optimizer.py` | 主优化器（预测+报童+约束） | ★★★ |
| `quant_engine/forecasting/gradient_boosted.py` | LightGBM/XGBoost 分位数回归 | ★★★ |
| `quant_engine/service.py` | 对外 Facade + 降级包裹 | ★★★ |
| `quant_engine/domain/value_objects.py` | CostParameters（Cu/Co/临界比率） | ★★ |
| `quant_engine/training/trainer.py` | TimeSeriesSplit 训练器 | ★★ |
| `quant_engine/fallback/safety_stock.py` | 安全库存兜底 | ★★ |
| `scripts/train_model.py` | 模型训练脚本 | ★★ |
| `scripts/backtest.py` | 回测脚本 | ★★ |
| `scripts/demand_uncensor.py` | 需求还原（缺货检测+真实需求估算） | ★★★ |
| `scripts/quantile_compare.py` | 多分位数模型训练+利润对比 | ★★ |
| `scripts/generate_chinese_report.py` | 中文可视化董事会报告生成 | ★★ |
| `tools/fetch_weather.py` | Open-Meteo 天气抓取 | ★ |

## 五、数学核心

### 报童模型（Newsvendor Model）

```
最优订货量 q* = F⁻¹(Cu / (Cu + Co))

Cu = 缺货成本 = (售价 - 成本) + 客户流失惩罚
Co = 滞销成本 = (成本 - 残值) + 持有成本

临界比率 = Cu / (Cu + Co)
  > 0.5 → 缺货比滞销贵 → 多订（饮料、主食）
  < 0.5 → 滞销比缺货贵 → 少订（生鲜、面包）
```

### 直接分位数回归（替代传统两步法）

```
传统两步法：预测需求分布 → 求逆 CDF → 得到 q*
我们的方法：直接训练 LightGBM 在临界比率分位数处预测 → q* = 预测值

优势：不需要假设需求分布形态（正态/泊松），直接数据驱动
```

## 六、数据清单

> 公开版本说明：原始数据来自一家合作连锁超市，授权已收回。数据、门店名称、地点和经营数字都不公开，下面只保留数据的结构和规模。

### 6.1 已有数据文件（不在仓库中）

| 来源 | 内容 | 规模 |
|------|------|------|
| POS 交易数据（科脉），3 家门店 | 按门店分目录的 .xls 导出 | 292 个文件，约 1400 万行交易记录，聚合为 2024 年 680 万行、2025 年 710 万行日度 SKU 数据，约 4.4 万个 SKU |
| ERP 数据（管家婆），全公司范围 | 商品主数据（商品名、条码、规格、品类）；按 SKU 的年度进货和销售统计 | — |
| 门店级进货汇总 | 三店各自的年度进货总量和总金额，没有 SKU 明细 | — |
| 天气 | 两个城市 2024–2026 年日度天气，Open-Meteo API | — |
| 节假日 | 代码生成，17 个节日，2024–2050，chinese-calendar + lunardate | — |

### 6.2 缺失的数据

| 数据 | 为什么需要 | 影响 |
|------|-----------|------|
| 三店按 SKU 的分店进货明细 | 做 SKU 粒度的精准利润对比，而非 COGS ratio 近似 | 当前只能用全店统一 COGS 率估算模型成本 |
| 年初库存快照 | 量化旧库存对当年利润的补贴 | 当年进货量低于销量，差额来自上一年的库存 |
| 真实废损/报损记录 | 校准品类损耗率（目前用关键词分类：生鲜 50%、短保 30%、常温 0%） | 损耗率是假设值，不是实测 |
| 分店逐日/逐 SKU 进货记录 | walk-forward 评估的对比基线 | 目前只能做年度汇总对比 |
| 供应商政策（MOQ、订货周期、整车约束） | Step-10 的约束优化 | 当前报童模型无约束 |

### 6.3 中间产物

| 文件 | 说明 |
|------|------|
| `data/raw/raw_pos.parquet` | 首次读 .xls 后的缓存（1400 万行，264 MB） |
| `data/features/features_train.parquet` | 2024 全年特征（训练集，680 万行） |
| `data/features/features_oos.parquet` | 2025 全年特征（测试集，710 万行） |
| `models/step3_results.joblib` | 9 个分位数模型（q=0.50 ~ 0.95） |
| `backtests/report/board_report.html` | 中文可视化董事会报告 |

## 七、技术栈

| 层 | 技术 |
|----|------|
| 预测引擎 | LightGBM (分位数回归), XGBoost (备选) |
| 数据处理 | pandas, numpy, scipy |
| 特征存储 | Parquet (列式压缩) |
| 节假日 | chinese-calendar (官方假日+调休), lunardate (农历) |
| API | FastAPI, pydantic v2 |
| 认证 | JWT (python-jose) |
| 容器化 | Docker (多阶段构建), docker-compose |
| 测试 | pytest (77 个测试) |
| 可视化 | plotly (交互图表), HTML 报告 |
| 外部数据 | Open-Meteo (天气) |
