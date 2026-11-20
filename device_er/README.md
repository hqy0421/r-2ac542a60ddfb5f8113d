# device_er — 设备档案全局一致实体消解

来自不同系统（erp / mes / iot）的设备档案，逐条按相似概率合并会产生矛盾
（传递性冲突、约束冲突）。本项目实现**全局最低成对期望风险的一致实体划分**，
并真正把结果落到业务库与可重开的代（generation）中。

## 流水线

1. **原生特征提取**（`device_er/features.py`）：用 tsfresh 从每条档案的遥测
   时序（vibration / temperature / current / pressure）真实提取特征
   （`CURATED_FC_PARAMETERS`，确定性）。提取失败 → `ExtractionError`，保旧。
2. **成对匹配概率**（`device_er/pairwise.py` + `device_er/model.py`）：加载
   **已拟合**的成对模型（`model/pairwise_model.pkl` + 自描述
   `model_card.json` 兜底），对全部档案对计算 P(同一实体)。模型缺失/损坏 →
   `ModelError`，保旧。
3. **全局优化**（`device_er/optimize.py`）：以误合并成本 `C_FM`、漏合并成本
   `C_DM` 最小化期望成对风险
   `R(x)=Σ[x_ij(1-p_ij)C_FM + (1-x_ij)p_ijC_DM]`，x 满足三角不等式（传递性）
   与必合/必分硬约束 —— scipy/HiGHS MILP 精确求解（小规模有枚举兜底），
   **不是阈值连通**。阈值连通分量仅作为基线写进风险解释。约束不可行或求解
   失败 → `OptimizationError`，保旧。
4. **主档合并**（`device_er/master.py`）：实体 ID = 最小成员 ID；每个属性按
   来源优先级取首个非 NULL 值 —— `0`、`FALSE`、`''` 都是值，会被保留
   （`is_null` 只认 None/NaN/NA/NaT）。
5. **工单重绑**（`device_er/rebind.py`）：既有维修工单的 `entity_id` 实际改
   绑到所属实体（最小成员 ID）；源档案已消失 → 置 NULL；其余业务列不动。
6. **原子代保存**（`device_er/storage.py`）：完整代写入
   `generations/.tmp-*` → 校验 SHA-256 → 原子改名 → 单个 SQLite 事务内
   （重绑工单 + 注册代 + 切换 current）。任何一步失败，旧代与业务数据原样
   保留。`reopen()` 校验清单后完整重开。

## 使用

```bash
export PYTHONPATH=/workspace/device_er:/workspace   # 后者提供 tsfresh 源码
python -m device_er --root /workspace/device_er seed --force   # 建演示业务库
python -m device_er --root /workspace/device_er train          # 拟合并持久化成对模型
python -m device_er --root /workspace/device_er run            # 完整跑一代（输出报告 JSON）
python -m device_er --root /workspace/device_er show           # 重开当前代
python -m device_er --root /workspace/device_er list           # 代登记簿
```

退出码：`0` 成功；`2` 流水线失败（已保旧，stderr/stdout 给出失败域）。

## 目录

```
data/device_er.db          业务库（档案/遥测/工单/约束/成本/优先级 + 代登记）
model/pairwise_model.pkl   已拟合成对模型（+ .card.json 自描述副本）
generations/gen_XXXX/      完整可重开代：features.csv、pair_probabilities.csv、
                           partition.json、master_records.*、work_orders_after.csv、
                           work_order_changes.csv、risk_pairs.csv、report.json、
                           config.json、model_card.json、model_copy.pkl、manifest.json
generations/CURRENT        当前代指针（库内 meta 表为权威）
```

## 测试

```bash
PYTHONPATH=/workspace/device_er:/workspace python -m pytest tests/ -q
```

36 项：NULL 语义（0/FALSE/'' 保留）、MILP 与枚举一致、链式概率下全局最优 ≠
阈值连通、约束不可行报错、重绑/置 NULL、代保存-重开-篡改检测、四类失败域
（提取/模型/优化/保存）保旧、端到端重跑幂等。
