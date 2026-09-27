# 固定速率陀螺仪 Allan 误差分解（离线/分批可复用模块）

对定频采样的单轴角速率记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解。统计尺度可在开始时预先选定，之后逐样本或逐批追加；同一份数据无论如何
分批，结果都与一次性离线计算一致。

> 拟合得到的斜率只是对**本批离线数据** log-log 形态的经验描述，不命名、不暗示
> 任何临床结论或硬件指标保证，本模块也不宣称完整兼容任何行业标准；它只实现
> 本题契约内的数学子集。

## Python 版本与依赖

- Python **3.14.7**（仅使用标准库：`math`、`dataclasses`、`collections.abc`）。
- 安装 Python 后**无需安装任何第三方依赖**，无需网络/账号/硬件/数据库/Docker。

## 运行命令

```bash
# 必要测试（独立参考，不调用被测核心产生期望值）
python -m unittest discover -s tests -v

# 固定输入演示（确定性合成数据；实测约 0.1 秒内完成，远小于 8 秒预算）
python demo.py
```

## 公开接口

```python
from allan_deviation import analyze, StreamingAllan
result = analyze(samples, dt, cluster_sizes)

stream = StreamingAllan(dt, cluster_sizes)
result = stream.append(first_batch)
result = stream.append(next_batch)  # 也可传入单个 int/float 样本
result = stream.snapshot()
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
| `fit` | `FitResult`：`segments`、`breakpoints`、`slopes`、`intercepts`、`sse`、`rmse`、`residuals`、`fitted_ms`、`segment_count`、`objective`；流式追加时若合格点不足 3 个则为 `None` |
| `dt` / `n_samples` | 回显校验后的 dt（转为 float）与样本数 |

`StreamingAllan(dt, cluster_sizes)` 在构造时固定并校验全部统计尺度；后续不能增删
尺度，保证每次返回的点顺序稳定。`append(samples)` 的参数可以是一个有限 `int/float`
（`bool` 拒绝），也可以是一个非空有限序列；累计样本数仍受 8192 上界约束。
非法批次会在任何状态更新前整体拒绝，不会留下部分追加。

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

## 算法与设计取舍

1. **重叠 Allan 方差**：离线和分批共用同一状态机。状态以首个样本为参考消除
   恒定偏置；每个尺度维护三点组合 `B[k]=Σw[k+m:k+2m]−Σw[k:k+m]`，新样本用
   `B[k+1]=B[k]+w[k+2m]−2w[k+m]+w[k]` 更新，并每 32 个差分从保留历史用
   `math.fsum` 精确重建，避免长序列递推误差累积。平方和用补偿求和，归一化用
   无偏因子 `1/(2(count−1))`。
2. **跨批窗口与历史**：新差分只在窗口完整时计数一次；批次边界不特殊处理，因此
   不会遗漏或重复。对象只保留最长尺度计算下一个三点组合所需的
   `2*max(m)+1` 个参考样本，以及每个尺度的当前组合、计数和平方和。
3. **拟合点筛选**：差分样本数 `>= 20` 且方差严格为正。较长尺度样本不足时其
   `variance/adev/count/eligible_for_fit` 独立表示，不影响其他尺度。常量信号（方差精确
   为 0，如全 0 或全 3.5）在 log 域无定义，不参与拟合。
4. **分段直线**：点按 `m` 去重升序后，枚举全部**连续**切分（段间无重叠、无
   缺口），段数 1..3，每段至少 3 点；段内为闭式 OLS，全局用**动态规划**
   `dp[seg][b] = min_a dp[seg−1][a−1] + SSE(a,b)` 求最小原始 SSE，再以
   `SSE + (p−1)·SEGMENT_PENALTY`（罚项常量 `SEGMENT_PENALTY = 1.0`，
   单位为 log(adev) 残差平方）选择段数。并列时取段数更少者。
5. **搜索预算 vs 无解**：DP 是对全部合法切分的精确穷举，预算常量
   （`_SEARCH_BUDGET`，8·K² 量级）取公开输入边界理论求值上界之上，因此对
   合法输入**不可达**；若预算耗尽抛 `SearchBudgetExhaustedError`（无法证明
   最优），与已证明不存在合法切分的 `FitInfeasibleError` 严格区分。
6. **无部分变更**：`analyze` 为纯函数；`StreamingAllan.append` 先整体校验并
   检查累计上界，通过后才更新状态。离线拟合不足 3 点仍抛
   `FitInfeasibleError`，其 `.points` 附带逐 m 统计；流式快照则返回 `fit=None`。

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
- 以上均继承 `AllanError`。

## 浮点误差口径（不以宽容差掩盖边界）

- 全部计算为 IEEE-754 binary64（Python float）。
- 测试中核心与**朴素双重循环窗口求和参考**（不去偏置、直接累加）的方差
  相对误差要求 `< 1e-12`（数据量级 O(1)、N≤300）；这是具体测得的数值口径，
  不是对任意量级输入的通用保证——量级接近 `1e308` 或噪声相对基线接近
  `2^-53` 的输入即使去偏置也无法恢复被 binary64 抹掉的低位。
- log-log 拟合目标值与**枚举全部切分的独立参考**要求绝对差
  `< 1e-9·max(1,|参考值|)`；断线下标要求完全一致（构造数据无并列）。
- `variance` 精确为 0 的判定来自浮点精确抵消；非二进制可精确表示的常量
  （如重复的 `0.1`）理论上可能留下 `~1e-16` 相对量级残差，这是 binary64
  表示本身的限制，模块不做"接近 0 就当 0"的隐式钳制。

## 已知限制

- 只处理单轴、定频序列；支持离线一次输入和固定尺度分批追加，但不做多轴相关、
  不做去趋势、不加窗、不做置信区间或噪声系数换算。
- `StreamingAllan` 的统计尺度在构造时固定；追加过程中不能新增或删除尺度。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时计算量为 O(N·K)（约 6.7×10⁷ 次
  差分级运算），属接口上界规模；短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（核心算法与全部数据结构/错误类型）。
- `tests/test_allan.py`：必要测试；参考实现（朴素窗口求和、itertools
  枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心；另覆盖任意分批
  与一次性离线结果一致、跨批窗口、非法批次原子性和未成熟尺度。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，打印分解结果、
  经验区间、理论交叉点对照，并真实触发 NaN、bool 类型拒绝与短数据无解边界。
