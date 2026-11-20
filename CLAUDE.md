# RetailLab — AI 协作规范

## 环境说明

- **WSL 仅用于 tmux**，不要在 WSL 里跑 Python/项目命令
- 项目运行环境是 **Windows**，使用 Windows 上的 Python 解释器
- 路径 `/mnt/c/Users/n1rva/projects/retaillab` 对应 Windows 路径 `C:\Users\n1rva\projects\retaillab`

## 铁律（每次会话必须遵守）

1. **每次操作前必须先读 `plan.md`**，确认当前进度和下一步任务
2. **每完成一个 Step 或子任务后，立即更新 `plan.md` 的进度标记**
   - `[PENDING]` → `[IN_PROGRESS]` → `[DONE]`
3. **不允许跳步**：必须按 plan.md 中的 Step 顺序执行，除非用户明确要求
4. **新会话开始时**：先读 `CLAUDE.md` → 再读 `plan.md` → 确认上下文后再动手

## 项目架构红线

### 模块隔离（违反即回退）
- `quant_engine/` **绝不**导入 `api_gateway/` 或 `data_pipeline/`
- `data_pipeline/` **绝不**导入 `quant_engine/`
- `api_gateway/` 只能通过 `quant_engine.service` (Facade) 访问引擎
- 模块间通信只通过 numpy arrays、dataclasses、Parquet 文件

### 安全红线
- API **绝不**暴露模型权重、特征重要性、训练数据
- `.env` 文件**绝不**提交到 git
- JWT secret **绝不**硬编码

### 数学红线
- 所有时序特征必须 `.shift(1)` 防数据穿越
- 训练必须用 `TimeSeriesSplit`，禁止普通 `KFold`
- 报童模型临界比率 `Cu/(Cu+Co)` 是底层逻辑，不可绕过

### 防御性工程红线
- `quant_engine/service.py` 必须包裹 try-except 降级到安全库存
- `api_gateway/routes/` 必须有二级 try-except 兜底
- 每笔补货决策必须写入 telemetry 日志
- 概念漂移超 2σ 必须触发告警

## 代码规范

- Python 3.12+
- 所有函数必须有 Type Hints
- 公开 API 必须有 Docstrings
- 使用 ruff 格式化，mypy 类型检查
- 测试覆盖核心数学和所有防御模块

## 关键文件路径

| 文件 | 职责 |
|------|------|
| `plan.md` | 项目进度跟踪（每次必读必更新） |
| `CLAUDE.md` | 本文件，AI 协作规范 |
| `quant_engine/optimization/inventory_optimizer.py` | 核心优化器 |
| `quant_engine/optimization/newsvendor.py` | 报童模型数学 |
| `quant_engine/service.py` | 对外 Facade + 降级包裹 |
| `quant_engine/telemetry/decision_logger.py` | 决策日志 |
| `quant_engine/monitoring/drift_detector.py` | 漂移监控 |
| `quant_engine/fallback/safety_stock.py` | 安全库存兜底 |
| `data_pipeline/feature_engineering/leakage_guard.py` | 穿越防护 |

## 日志体系（三个文件，各有分工）

| 文件 | 类型 | 用途 |
|------|------|------|
| `backtests/RESEARCH_LOG.md` | **手写** | 决策记录：为什么改了什么、发现了什么问题、怎么解决的。**面试展示用**，要体现逐步迭代和思考成长。 |
| `backtests/training_log.csv` | **自动** | 每次跑 `train_model.py` 自动追加：参数、MAE、RMSE、coverage、top特征、训练时间。 |
| `backtests/backtest_log.csv` | **自动** | 每次跑 `backtest.py` 自动追加：MAE、缺货率、成本、Cu/Co。 |

### RESEARCH_LOG 写作规范
- 每次改动后由 AI 帮用户总结，语言要**干练、专业**
- 结构：**改了什么 → 为什么改 → 结果/影响**
- 要体现用户的思考过程和判断力（例如"用户发现 total_amount 存在数据穿越"而不是"删除了 total_amount"）
- 这是用户的作品集，不是 AI 的工作日志

## 提交规范

- 每完成一个 Step 做一次 git commit
- commit message 格式：`step-N: 简要描述`
- 不提交 `.env`、模型权重文件、`__pycache__/`
