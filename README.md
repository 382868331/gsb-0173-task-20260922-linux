# 固定速率陀螺仪 Allan 误差分解（离线可复用模块）

对定频采样的单轴角速率离线记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。

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
from allan_deviation import analyze
result = analyze(samples, dt, cluster_sizes)
```

另有**分批/流式**接口（同一套数学，固定尺度、持续接收数据）：

```python
from allan_deviation import streaming_allan
acc = streaming_allan(dt, cluster_sizes)   # 构造时预先选定全部统计尺度
snap = acc.append(batch)                   # 每追加一批，返回截至当前的统计
snap = acc.snapshot()                      # 只读当前结果，不改变状态
acc.close()                                # 可选：结束后拒绝再 append
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

流式接口返回 `StreamingAllanResult`：

| 字段 | 含义 |
|---|---|
| `points[i]` | 与构造时 `cluster_sizes` 一一对应（含重复、原顺序）的 `StreamingAllanPoint` |
| `fit` | 截至当前数据的 `FitResult`；合格拟合点不足 3 个时为 `None`（原因见 `fit_infeasible_reason`），逐尺度统计照常返回 |
| `fit_infeasible_reason` | `fit is None` 时的原因字符串，否则为 `None` |
| `n_samples` / `dt` | 截至当前已接收的样本总数与采样周期 |
| `retained_samples` / `retained_limit` | 当前/上限保留历史样本数，上限恒为 `2*m_max-1`（`m_max=max(cluster_sizes)`） |

`StreamingAllanPoint` 与 `AllanPoint` 字段一致（`m/tau/variance/adev/count/
eligible_for_fit`），另含：

- `computable`：该尺度当前是否可做无偏方差估计（`count >= 2`）。样本不够时
  长尺度**单独**标为 `computable=False`、`variance/adev=None`，其他尺度不受
  影响；`count` 与 `tau` 始终返回。
- `n_samples`：该快照对应的累计样本总数。

流式语义（跨批次窗口）：

- 第 `k` 个相邻块均值差分（需要样本 `k .. k+2m-1`）只在其**末端样本到达的
  那一批**被计入一次；早到的样本仅作为历史保存，因此跨批次窗口**不遗漏、
  不重复计数**，且 `count(m)=max(0, N-2m+1)` 与分批方式无关。
- 内部只保留最近 `2*m_max-1` 个去偏置样本（任何未来窗口差分只需这些），
  内存为 `O(m_max)`，与已接收样本总数无关；平方和用 Neumaier 补偿求和。
- 同一数据无论切成多少批喂入，结果都与一次性 `analyze` 一致（见误差口径）。
- `append` 先整体校验批次、再把本批新差分全部算出，任一步失败则累加器
  状态完全停留在该批之前（无部分变更）。`close()` 后 `append` 抛
  `StreamingClosedError`，`snapshot()` 仍可读。
- 流式总样本数**没有** 8192 上限（8192 仅为离线接口的一次性输入边界）；
  每批必须非空，`dt` 与尺度在构造时固定、之后不可更改。

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
- `StreamingClosedError`（同时是 `RuntimeError`）：对已 `close()` 的流式
  累加器继续 `append`。流式批次非法（空批、NaN/Infinity、bool/str 等）复用
  `AllanTypeError`/`AllanValueError`，错误下标定位到 `batch[i]`。
- 以上均继承 `AllanError`。

## 浮点误差口径（不以宽容差掩盖边界）

- 全部计算为 IEEE-754 binary64（Python float）。
- 测试中核心与**朴素双重循环窗口求和参考**（不去偏置、直接累加）的方差
  相对误差要求 `< 1e-12`（数据量级 O(1)、N≤300）；这是具体测得的数值口径，
  不是对任意量级输入的通用保证——量级接近 `1e308` 或噪声相对基线接近
  `2^-53` 的输入即使去偏置也无法恢复被 binary64 抹掉的低位。
- log-log 拟合目标值与**枚举全部切分的独立参考**要求绝对差
  `< 1e-9·max(1,|参考值|)`；断线下标要求完全一致（构造数据无并列）。
- 流式接口与离线 `analyze`、以及**同一数据不同分批方式**之间，方差相对差
  `< 1e-12`（实测 O(1) 数据通常 `1e-14~1e-13`，见 demo 打印）；差异仅来自
  平方和累加顺序（离线 `fsum`、流式 Neumaier 补偿求和）。计数 `count`、
  `computable`、拟合断线下标要求与离线结果完全一致。
- `variance` 精确为 0 的判定来自浮点精确抵消；非二进制可精确表示的常量
  （如重复的 `0.1`）理论上可能留下 `~1e-16` 相对量级残差，这是 binary64
  表示本身的限制，模块不做"接近 0 就当 0"的隐式钳制。

## 已知限制

- 只处理单轴、定频序列；离线 `analyze` 限一次性 8192 点，流式
  `streaming_allan` 支持任意总样本数（内存 `O(m_max)`）；不做多轴相关、
  不做去趋势、不加窗、不做置信区间或噪声系数换算。
- 流式尺度与 `dt` 在构造时固定；不支持运行中增删尺度或变速采样。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时计算量为 O(N·K)（约 6.7×10⁷ 次
  差分级运算），属接口上界规模；短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（核心算法与全部数据结构/错误类型）。
- `tests/test_allan.py`：离线核心必要测试；参考实现（朴素窗口求和、itertools
  枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心。
- `tests/test_streaming.py`：流式接口必要测试；方差期望值复用朴素窗口参考与
  离线 `analyze`，并用多种分批划分交叉验证跨批窗口不遗漏/不重复、历史保留
  裁剪、长尺度单独不可计算、非法批次无部分变更与 close 语义。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，先打印离线分解、
  经验区间、理论交叉点对照并触发 NaN/bool/短数据边界，再演示流式分批
  append（长尺度从不可算到可算、历史保留、与离线一致性）及非法批次/close
  边界。
