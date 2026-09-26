# 固定速率陀螺仪 Allan 误差分解与离线温度补偿

对定频采样的单轴角速率离线记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。

本版本新增**离线温度补偿迭代**：在调用方指定的训练区间拟合常数、一阶和二阶
温度项，对完整陀螺序列补偿后，分别报告训练区间与留出区间的 Allan 结果。

> 拟合得到的斜率只是对**本批离线数据** log-log 形态的经验描述，不命名、不暗示
> 任何临床结论或硬件指标保证，本模块也不宣称完整兼容任何行业标准；它只实现
> 本题契约内的数学子集。

## Python 版本与依赖

- Python **3.14.7**（仅使用标准库：`math`、`dataclasses`、`collections.abc`）。
- 安装 Python 后**无需安装任何第三方依赖**，无需网络/账号/硬件/数据库/Docker。

## venv 准备与运行命令

```bash
# Linux 原生 Python 3.14.7；仅标准库，venv 中无需 pip install。
python3.14 -m venv .venv
source .venv/bin/activate
python --version  # 应显示 Python 3.14.7

# 必要测试（独立参考，不调用被测核心产生期望值）
python -m unittest discover -s tests -v

# 温度补偿现场计算演示（确定性合成数据；实测约 0.1 秒内完成，预算 90 秒）
python demo_iteration.py

# 旧基线演示仍可继续运行
python demo.py
```

## 公开接口

```python
from allan_deviation import analyze
result = analyze(samples, dt, cluster_sizes)

from allan_deviation import analyze_with_temperature, fit_temperature_compensation
fit = fit_temperature_compensation(samples, temperatures, train_range, timestamps=None)
result = analyze_with_temperature(
    samples, temperatures, dt, cluster_sizes, train_range, timestamps=None
)
```

- `samples`：定频单轴角速率序列，长度 **1..8192**（接口上限），元素为
  `int`/`float` 且全部有限；`bool`、`str`、`complex` 等按类型错误拒绝，
  `NaN`/`±Infinity` 按取值错误拒绝。
- `dt`：采样周期（秒），`int`/`float`、有限且 `> 0`；`bool` 拒绝。
- `cluster_sizes`：聚类大小 `m`（样本数）的有限序列（list/tuple/range），
  长度 **1..8192**；每个元素必须是**真正的 `int`**（`bool`、`1.0` 拒绝）
  且 `m >= 1`，且 `m*dt` 必须可表示为有限 float。允许**重复**与**非单调**：
  输出点顺序与输入一致；拟合前自动按 `m` 去重并升序排列。

返回 `AllanResult`（不可变 dataclass）：

| 字段 | 含义 |
|---|---|
| `points[i]` | 与输入 `cluster_sizes` 一一对应的 `AllanPoint(m, tau, variance, adev, count, eligible_for_fit)` |
| `fit` | `FitResult`：`segments`、`breakpoints`、`slopes`、`intercepts`、`sse`、`rmse`、`residuals`、`fitted_ms`、`segment_count`、`objective` |
| `dt` / `n_samples` | 回显校验后的 dt（转为 float）与样本数 |

`AllanPoint`：

- `tau = m * dt`；
- `count = max(0, N - 2m + 1)`：**实际**相邻等长块均值差分样本数；
- `variance = Σ d_k² / (2·(count−1))`，`d_k = mean(w_{k+m}) − mean(w_k)`，
  `count < 2` 时为 `None`（无法做无偏估计）；
- `adev = sqrt(variance)`；
- 仅 `count >= 20` 且 `variance > 0` 的点 `eligible_for_fit=True`。

`Segment`：`y = slope·x + intercept`（`x=log(tau)`，`y=log(adev)`），并带
`m_start/m_end`、`tau_start/tau_end`、`n_points`。`Breakpoint` 定义为右侧段
第一个拟合点（`m`、`tau`）。`residuals` 与 `fitted_ms` 对齐，为
观测 `log(adev)` 减模型值。`objective = sse + (segment_count−1)·SEGMENT_PENALTY`。

### 温度补偿接口

- `temperatures`：与 `samples` 等长、逐样本对齐的有限实数温度序列；也接受由
  连续块组成的嵌套序列，边界只用于扁平化，不影响拟合结果。
- `train_range`：无时间戳时为闭区间整数下标，如 `(0, 699)`；提供
  `timestamps` 时为 `(start_timestamp, end_timestamp)`，两个时间戳都必须**精确
  存在**于输入中。不做最近邻、线性或任何其他插值。
- `analyze_with_temperature` 的训练区间必须是前缀或后缀，从而留出另一个连续
  区间；训练覆盖全部数据会被拒绝，不能把训练拟合改善当成独立验证。
- 常温数据退化为常数项拟合；非常温数据若训练行少于 3，或
  `[1,u,u²]` 设计矩阵列相关，抛 `TemperatureFitError`。
- 若训练区内温度与时间呈精确仿射关系，明确抛 `TemperatureFitError`，因为
  温度效应与时间漂移无法分离。

`fit_temperature_compensation` 返回 `TemperatureFit`：

| 字段 | 含义 |
|---|---|
| `center` / `scale` | 训练温度中心化值与尺度；`u=(temperature-center)/scale` |
| `coefficients` | `(c0,c1,c2)`，预测值为 `c0+c1*u+c2*u²`；常温时 `c1=c2=0` |
| `residuals` | 仅训练区间的独立残差，顺序对齐 `train_indices` |
| `compensated_samples` | 对**整段**输入执行 `sample-prediction` 后的序列 |
| `train_indices` | 实际训练闭区间下标 |
| `term_count` / `sse` / `rmse` | 实际使用项数及训练残差统计 |

温度最小二乘使用 Householder **QR 分解**求解，不构造 `XᵀX` 正规方程。

`analyze_with_temperature` 返回 `TemperatureAllanResult`：`temperature_fit`、
`training`、`holdout`、`dt`、`n_samples`。`training` 和 `holdout` 均为
`SplitAllanReport`，分别含 `original` 与 `compensated` 两个 `AllanSubsetReport`；
每个报告独立携带 Allan 点和分段拟合结果。若某区间长度不足以产生 3 个合格
log-log 点，该子报告的 `fit=None` 且 `infeasible_reason` 说明原因，不影响另一
区间或已完成的温度拟合。

## 算法与设计取舍

1. **重叠 Allan 方差**：先对样本做前缀累积和（此前用 `math.fsum` 精确均值
   去除恒定偏置——相邻块均值之差对整体加常数数学不变，去偏置可避免大基线
   信号在三项相减时灾难性抵消），每个差分用
   `d_k = (C[k+2m] − 2·C[k+m] + C[k]) / m` 得到；平方和用 `math.fsum`
   累加。归一化用无偏因子 `1/(2(count−1))`（有限记录的标准重叠估计写法）。
2. **拟合点筛选**：差分样本数 `>= 20` 且方差严格为正。常量信号（方差精确
   为 0，如全 0 或全 3.5）在 log 域无定义，不参与拟合。
3. **分段直线**：点按 `m` 去重升序后，枚举全部**连续**切分（段间无重叠、无
   缺口），段数 1..3，每段至少 3 点；段内为闭式 OLS，全局用**动态规划**
   `dp[seg][b] = min_a dp[seg−1][a−1] + SSE(a,b)` 求最小原始 SSE，再以
   `SSE + (p−1)·SEGMENT_PENALTY`（罚项常量 `SEGMENT_PENALTY = 1.0`，
   单位为 log(adev) 残差平方）选择段数。并列时取段数更少者。
4. **搜索预算 vs 无解**：DP 是对全部合法切分的精确穷举，预算常量
   （`_SEARCH_BUDGET`，8·K² 量级）取公开输入边界理论求值上界之上，因此对
   合法输入**不可达**；若预算耗尽抛 `SearchBudgetExhaustedError`（无法证明
   最优），与已证明不存在合法切分的 `FitInfeasibleError` 严格区分。
5. **无部分变更**：`analyze` 为纯函数，不改写输入、无全局状态；任何错误都
   先于返回抛出。`FitInfeasibleError.points` 附带已算好的逐 m 统计（异常即
   结果的一部分，不属于部分写入）。
6. **温度补偿**：先在训练温度上中心化并以训练温度跨度的一半作尺度；QR 在
   训练区间求系数，残差只用于报告训练误差，随后同一模型外推补偿完整序列。
   原数据和补偿数据分别送回同一 Allan 计算，训练与留出报告互不复用拟合结论。

## 错误语义

- `AllanTypeError`（同时是 `TypeError`）：类型非法，含对整数参数传 `bool`。
- `AllanValueError`（同时是 `ValueError`）：`NaN/Infinity`、`dt<=0`、
  空序列（`samples` 或 `cluster_sizes`）、长度超 8192、`m<1`、超大整数无法
  表示为有限 float 或 `m*dt` 溢出、样本量级使均值/方差计算溢出等；错误信息
  定位到参数名与下标（如 `samples[1]: ...`、`cluster_sizes[3]: ...`），
  不静默纠正、不返回 `inf/NaN` 结果。
- `FitInfeasibleError`：合格拟合点不足 3 个（常量零方差、短数据等），
  信息说明原因（证明无解）。
- `SearchBudgetExhaustedError`：搜索预算耗尽（公开边界内不会发生）。
- `TemperatureFitError`：温度模型不可识别或无法唯一拟合，包括训练行不足、
  温度多项式列相关、温度与时间在训练区内完全混淆、时间戳边界无法精确匹配。
- 以上均继承 `AllanError`。

## 浮点误差口径（不以宽容差掩盖边界）

- 全部计算为 IEEE-754 binary64（Python float）。
- 测试中核心与**朴素双重循环窗口求和参考**（不去偏置、直接累加）的方差
  相对误差要求 `< 1e-12`（数据量级 O(1)、N≤300）；这是具体测得的数值口径，
  不是对任意量级输入的通用保证——量级接近 `1e308` 或噪声相对基线接近
  `2^-53` 的输入即使去偏置也无法恢复被 binary64 抹掉的低位。
- log-log 拟合目标值与**枚举全部切分的独立参考**要求绝对差
  `< 1e-9·max(1,|参考值|)`；断线下标要求完全一致（构造数据无并列）。
- 温度测试用独立函数重新代回 `c0+c1*u+c2*u²` 核对每个训练残差和整段补偿值；
  测试中的多项式回代不调用被测核心，常温退化要求一次/二次系数严格为 0。
- “温度与时间完全混淆”按训练区内温度是否为时间的精确仿射关系判定；
  1e-12 相对容差只用于吸收同一离线输入上的 binary64 表示误差，不做插值。
- `variance` 精确为 0 的判定来自浮点精确抵消；非二进制可精确表示的常量
  （如重复的 `0.1`）理论上可能留下 `~1e-16` 相对量级残差，这是 binary64
  表示本身的限制，模块不做"接近 0 就当 0"的隐式钳制。

## 已知限制

- 只处理单轴、定频、离线序列；不做在线/流式估计、不做多轴相关、不做
  去趋势、不加窗、不做置信区间或噪声系数换算。
- 温度模型仅限常数、一阶、二阶项；不搜索更复杂温度滞后、交叉项或时间项模型。
- 时间戳只做逐点精确对齐；缺失、抖动到另一采样时刻或需要插值的输入一律拒绝。
- 留出结果只说明这一批离线合成/记录数据上的经验稳定性，不构成真实传感器
  精度、标定有效期、现场温漂或量产性能保证。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时计算量为 O(N·K)（约 6.7×10⁷ 次
  差分级运算），属接口上界规模；短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（Allan、分段拟合、QR 温度补偿及数据结构/错误）。
- `tests/test_allan.py`：必要测试；参考实现（朴素窗口求和、itertools
  枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心。
- `tests/test_temperature.py`：温度 QR 拟合测试；独立多项式回代、时间混淆反例、
  秩不足/数据不足、时间戳对齐、常温退化和切块输入等价。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，打印分解结果、
  经验区间、理论交叉点对照，并真实触发 NaN、bool 类型拒绝与短数据无解边界。
- `demo_iteration.py`：固定种子（20260926）现场生成已知温漂和确定性噪声，
  分别打印训练/留出、原数据/补偿数据 Allan，并真实触发温度-时间混淆及时间戳
  不匹配拒绝；无等待、无网络、无预录答案。
