# 固定速率陀螺仪 Allan 误差分解（allan_decomp）

离线、单轴、固定采样率角速率序列的**重叠 Allan 方差**计算，与 log10-log10
平面上的**连续分段直线**辨识（动态规划求全局最优分段 + 分段罚项）。

仅实现本题契约定义的有限数学子集：**不**宣称兼容 IEEE 1139 等任何完整行业
标准；输出的 log-log 斜率只是对给定离线数据的经验统计描述，**不**构成硬件
指标保证、噪声系数认证或任何临床结论，字段也不使用此类命名。

## 运行环境

- Python **3.14.7**（CPython，Linux 原生验证）；
- **仅标准库**（`math`、`dataclasses`、`typing`、测试用 `unittest`/`random`/
  `itertools`），安装 Python 后无需安装任何第三方依赖，无需网络；
- 核心、测试、demo 全部离线、确定性、无状态。

## 命令

```bash
# 必要测试（25 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（现场真实计算，本机实测约 0.2 秒；不使用 sleep 或预录结果）
python demo.py
```

终端命令仅用于启动测试与演示，没有 CLI 产品、服务或后台任务。

## 公开接口

```python
from allan_decomp import analyze

result = analyze(samples, dt, cluster_sizes, *,
                 segment_penalty=0.0, search_budget=50_000_000)
```

- `samples`：实数序列（`int`/`float`，元素全部有限；拒绝 `bool`、`NaN`、
  `±Infinity`、字符串、生成器等），长度 `1 <= N <= 8192`。
- `dt`：采样周期，正有限实数，拒绝 `bool`。
- `cluster_sizes`：正**精确整数**序列（拒绝 `bool`、`float`，包括整数值
  浮点），允许**重复**与**非单调**输入；m 必须 `>= 1`。
- `segment_penalty`：`>= 0` 的有限实数，每增加一段在 DP 目标上的罚项
  （量纲为 log10 残差平方），默认 0。
- `search_budget`：正精确整数，候选区间评估次数预算（拒绝 `bool`/`float`）。

### 返回：`AnalysisResult`（frozen dataclass）

| 字段 | 含义 |
| --- | --- |
| `n_samples`, `dt` | 实际接收的输入 |
| `curve` | 与 `cluster_sizes` **同序、含重复** 的每个 m 的 `ClusterPoint` |
| `fit_points` | 去重、按 m 升序、实际参与拟合的点（见下方资格规则） |
| `excluded` | 去重后未参与拟合的点与原因（按 m 升序） |
| `fit` | `PiecewiseFit`：断点、斜率、截距、残差与搜索信息 |
| `segments` | 物理量区间形式的分段（m/tau 起止、经验斜率、截距） |
| `breakpoints` | 段间连续连接点（位于左段最后一个拟合点的 x 坐标） |

`ClusterPoint`：`m`、`tau = m*dt`、`avar`、`adev = sqrt(avar)`、
`n_differences`。当 `2m > N` 时 `avar/adev` 为 `None`、`n_differences = 0`。

`PiecewiseFit`：`feasible`、`reason`、`starts/ends`（拟合点序列上的闭区间）、
`knot_x`（内部断点 x=log10(tau)）、`slopes`、`intercepts`（各段等效截距，
折线在断点处连续）、`partition_sse`（DP 目标的分段独立 OLS SSE 之和）、
`residual_sse`（最终连续折线在全部拟合点上的残差平方和）、`objective`
（= `partition_sse + (段数-1)*segment_penalty`）、`residuals`（逐点残差）、
`evaluations`、`budget`。

## Allan 方差定义与差分样本数

对每个簇长 m，在 N 点序列上取所有滑动位置 k（步长 1）：

```
d_k = mean(x[k+m : k+2m]) - mean(x[k : k+m]),   k = 0 .. N-2m
AVAR(m) = (1 / (2 n)) * sum_k d_k^2 ,            n = N - 2m + 1
```

`n_differences` 即该 m 的**实际差分样本数 n**（不是上界估计）；`2m > N` 时
无相邻等长块对，返回 `None`。块均值差经由一次累积和（cumsum）以 O(N) 求得，
全部簇长复用同一累积和；`d_k^2` 以 `math.fsum` 高精度串接。

拟合点资格（去重升序后）：

1. `n_differences >= 20`（否则记入 `excluded`，原因 `too_few_differences`；
   连一个差分都没有时为 `no_estimator`）；
2. `avar > 0`（零方差无法取 log10，原因 `zero_variance`）。

## 分段拟合

对资格点取 `(x, y) = (log10(tau), log10(avar))`：

- **最多 3 段、每段至少 3 个点**；
- 在全部合法连续分段上做**穷举动态规划**，段内为独立一元 OLS，目标为
  `SSE 之和 + (段数-1) * segment_penalty`——返回的是该离散分段空间内的
  全局最优，不是启发式近似；
- DP 为融合实现：每个候选区间 `[a, j]` 的 OLS SSE 只计算一次并同时用于
  2 段、3 段转移（区间 SSE 由二阶前缀矩 O(1) 给出）；
- 选定分段后，再以铰链基
  `y = c0 + c1 x + Σ_r d_r max(0, x − knot_r)` 的最小二乘求最终的
  **连续**折线（高斯消元解正规方程），因此段间斜率可变但在断点处 y 连续。
  注意连续约束模型的 `residual_sse` 可以略大于分段独立 OLS 的
  `partition_sse`，两者分别返回，这是定义差异而非错误；
- 确定性平局规则：罚项相同优先**段数更少**；段数相同优先**更靠左**的断点。

## 错误语义（非法输入不静默纠正）

所有异常继承 `AllanAnalysisError`（也是 `ValueError`）。
`InvalidInputError` 暴露 `parameter`、`index`、`reason`、`bad_value`，可定位
到参数与下标；例如 `cluster_sizes[1]` 为负、`samples[7]` 为 NaN。
除直接的输入有限性检查外，派生量（`m*dt`、累积和、差分平方和）一旦溢出为
非有限值也会在对应参数上抛出，而不是返回 `inf` 结果。

**搜索预算耗尽与证明无解严格分开：**

- 证明无解是**正常返回** `fit.feasible is False` 且 `reason ∈
  {"too_few_differences", "zero_variance", "insufficient_points"}`
  （资格点为 0 / 全零方差 / 正方差点不足 3 个）；这种路径不消耗搜索预算；
- `SearchBudgetExhausted` 只在调用方主动收紧 `search_budget`、可行问题尚未
  完成穷举时抛出；异常带 `evaluations/budget/n_points`。
  `analyze` 是无状态纯计算，失败时不产生任何部分输出或外部变更。

默认预算 `50_000_000` 覆盖全部合法输入：N≤8192 时拟合点
`n ≤ (N−19)//2 = 4086`，融合 DP 评估数 `n + n(n−5)/2 < 8.4e6`。

## 浮点误差口径（不以宽容差掩盖边界错误）

- 计算使用 IEEE-754 binary64；块和为普通双精度累加（最大 8192 项），
  差分平方和用 `math.fsum` 串接。对 `|x| ≲ 1` 的典型输入，AVAR 与朴素双重
  循环参考的相对偏差约为 1e-15 量级；测试以 **相对误差 1e-11**（另加
  1e-14 绝对地板）核对，这是远宽于实际误差的核对口径，不代表结果被该容差
  定义。
- log10 变换、OLS 与铰链拟合的误差为双精度常规量级；测试对 DP 的
  目标/SSE 与独立枚举参考按相对 1e-9（绝对 1e-12）核对，断点连续性按
  绝对 1e-12 核对。
- 数学上非负的量（SSE）若因消去得到严格为负的微小值，钳为 0。
- 输入量级极端（接近 binary64 上溢边界）时行为以“派生量必须有限”为准，
  见错误语义；本库不承诺次正规数/极端缩放下的特定相对精度。

## 设计取舍与已知限制

- 只处理**单轴、固定 dt** 的角速率采样；不做多轴相关、插值重采样或
  非均匀采样。
- 重叠实现滑动步长为 1；m 集合以调用方给出去重后的值为准，不自动生成
  对数等距网格。
- AVAR 仅使用“相邻”块对（overlapped Allan variance），不实现修正 Allan
  方差、总方差或置信区间/自由度修正。
- 分段数上限 3、每段最少 3 点是本题硬契约；断点只能落在已有的拟合点
  x 坐标上（离散分段空间），不做连续 x 上的断点优化。
- `residual_sse` 对应连续铰链模型；`partition_sse`/`objective` 对应 DP 的
  分段独立 OLS，二者有意分别报告。
- 性能：DP 为 O(n²) 纯 Python。接口上限（N=8192、m=1..4086）的最坏输入在
  参考机上约 6 秒以内；演示与测试规模均远小于此。
- 无并发、无 I/O、无全局可变状态；重复结果在同一次调用中经缓存复用。

## 测试与参考的独立性

`tests/test_allan.py` 中的期望值来自两个**不调用被测核心**的独立参考：

- `naive_avar`：朴素双重循环窗口重算每个块和；
- `enumerate_segments` + `ref_ols`：枚举全部合法分段边界并独立最小二乘。

被测核心自身的连续性、残差重算、预算计数等只作为补充断言。覆盖：常量零
方差、重复/非单调 m、短数据的各类无解原因、类型与范围拒绝（含 bool、
NaN/Inf、上溢）、预算耗尽 vs 证明无解、罚项行为，以及合成白噪声
（经验斜率 ≈ −1）与白噪声+斜坡长簇区间（经验斜率 ≈ +2）的区间辨识。

## 演示数据

`demo.py` 用固定种子 `20260922` 的标准库 `random` 生成 4096 点
“高斯白噪声 + 线性速率斜坡”，打印 AVAR 曲线、分段斜率/断点与残差，并现场
触发：一个 `InvalidInputError`（NaN）、一个证明无解返回（常量信号）、一个
`SearchBudgetExhausted`（收紧预算）。全部为仓库内确定性生成，不依赖附件、
网络或其他题目代码。
