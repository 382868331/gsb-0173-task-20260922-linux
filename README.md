# 陀螺仪 Allan 误差分解（离线可复用模块）

对单轴角速率离线记录计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。支持两种时间轴：

- **定频模式**：只给采样周期 `dt`，聚类大小以样本数给出（原有入口，不变）；
- **时间戳模式**（本次新增）：给每个样本的实际**时间戳**与**有效标志**，
  聚类大小以**时窗长度（秒）**给出；按样本实际落入的固定端点时窗统计相邻
  窗均值差，覆盖不足的窗对记为**缺失**。时间戳严格等间隔且全部有效时，
  结果与定频模式一致。

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

# 固定输入演示（确定性合成数据；实测 0.1 秒内完成，远小于 90 秒预算）
python demo.py
```

## 公开接口

### 定频模式（原有调用，保持可用）

```python
from allan_deviation import analyze
result = analyze(samples, dt, cluster_sizes)
```

- `samples`：单轴角速率序列，长度 **1..8192**（接口上限），元素为
  `int`/`float` 且全部有限；`bool`、`str`、`complex` 等按类型错误拒绝，
  `NaN`/`±Infinity` 按取值错误拒绝。
- `dt`：采样周期（秒），`int`/`float`、有限且 `> 0`；`bool` 拒绝。
- `cluster_sizes`：聚类大小 `m`（样本数）的有限序列（list/tuple/range），
  长度 **1..8192**；每个元素必须是**真正的 `int`**（`bool`、`1.0` 拒绝）
  且 `m >= 1`，且 `m*dt` 必须可表示为有限 float。允许**重复**与**非单调**：
  输出点顺序与输入一致；拟合前自动按 `m` 去重并升序排列。

### 时间戳 + 有效标志模式（新增）

```python
result = analyze(samples, timestamps=ts, valid=flags,
                 cluster_durations=durations)
```

- `timestamps`：每个样本的时间戳（秒），与 `samples` 等长、`int`/`float`
  有限且**严格递增**；`bool` 拒绝。
- `valid`：可选，与 `samples` 等长的有效标志，元素为 `bool` 或整数 `0/1`；
  缺省视为全部有效。无效样本既不进入时窗均值，也不参与标称周期估计。
- `cluster_durations`：聚类时窗长度（秒）的有限序列，长度 **1..8192**，
  元素 `int`/`float`、有限且 `> 0`；允许重复、非单调。输出点的
  `AllanPoint.m` 即窗长（float 秒），`tau == m`。
- 两种模式的参数**不得混用**（同时给 `dt`/`cluster_sizes` 与
  `timestamps`/`cluster_durations` 会被明确拒绝）。

时间戳模式返回的 `AllanResult` 额外字段：`mode="timestamp"`、
`n_valid`（有效样本数）；`dt` 回显**标称周期**（相邻有效样本时间间隔的
中位数，有效样本不足 2 个时为 `NaN`——此时没有任何可考察窗对），定频
模式下 `mode="fixed"`、`n_valid == n_samples`。

**固定端点规则（确定、不随数据改变）**：

1. 标称步长 `s` = 相邻有效样本时间间隔的中位数；`t0` = 首个有效样本时间戳。
2. 每个有效样本按 `q=(t−t0)/s` 归到整数网格单元（与整数偏差在 `1e-9`
   相对容差内时吸附，吸收 binary64 时间戳的舍入；否则向下取整），单元号
   非递减。
3. 窗长 `w` 折合 `M=w/s` 个单元；第 `g` 对时窗为单元区间
   `[g, g+M)` 与 `[g+M, g+2M)`（左闭右开，边界吸附到整数网格线），
   即两个**等长且起点相隔一个窗长**的时窗；`g=0,1,2,...`。
4. 时窗对并集越过末端网格（`g+2M > cmax+1`，`cmax` 为最后有效样本单元）
   后不再考察；考察窗对总数为 `expected_count`。
5. 时窗内有效样本数必须 `>= need = ceil(0.5·M)`（至少 1，常量
   `MIN_WINDOW_FRACTION = 0.5`），否则该侧时窗缺失；窗对任一侧缺失即跳过，
   计入 `missing_count`。
6. 对双侧达标的窗对计算
   `d_g = mean([g+M,g+2M)) − mean([g,g+M))`，
   `variance = Σd²/(2(count−1))`；`count < 2` 时 `variance=None`
   （覆盖不足返回缺失，不抛异常、不补零）。

时间戳严格等间隔、全部有效且 `w` 为 `s` 的整数倍时，时窗与定频模式的
等长样本块逐块重合，两条代码路径给出相同的 `count` 与方差（差别仅
binary64 舍入级）。`w` 不是 `s` 整数倍时，跨格边界按同一吸附规则
（含下界、不含上界）处理。

### 返回结构

返回 `AllanResult`（不可变 dataclass）：

| 字段 | 含义 |
|---|---|
| `points[i]` | 与输入一一对应的 `AllanPoint(m, tau, variance, adev, count, eligible_for_fit, expected_count, missing_count)` |
| `fit` | `FitResult`：`segments`、`breakpoints`、`slopes`、`intercepts`、`sse`、`rmse`、`residuals`、`fitted_ms`、`segment_count`、`objective` |
| `dt` / `n_samples` | 定频模式回显 dt（float）；时间戳模式回显标称周期 |
| `mode` / `n_valid` | `"fixed"`/`"timestamp"`；有效样本数 |

`AllanPoint`：

- 定频模式 `tau = m*dt`；时间戳模式 `tau = m`（窗长秒数）。
- `count`：**实际**双侧达标的相邻窗均值差分样本数。
- `expected_count`：按固定端点规则考察的相邻窗对总数（含缺失）；
  定频模式恒等于 `count`（无缺失概念）。
- `missing_count = expected_count − count`：覆盖不足被跳过的窗对数；
  定频模式恒为 0。
- `variance = Σ d_k² / (2·(count−1))`，`count < 2` 时为 `None`；
  `adev = sqrt(variance)`。
- 仅 `count >= 20` 且 `variance > 0` 的点 `eligible_for_fit=True`。

`Segment`：`y = slope·x + intercept`（`x=log(tau)`，`y=log(adev)`），并带
`m_start/m_end`、`tau_start/tau_end`、`n_points`。`Breakpoint` 定义为右侧段
第一个拟合点（`m`、`tau`）。`residuals` 与 `fitted_ms` 对齐，为
观测 `log(adev)` 减模型值。`objective = sse + (segment_count−1)·SEGMENT_PENALTY`。

## 算法与设计取舍

1. **重叠 Allan 方差（定频模式）**：先对样本做前缀累积和（此前用 `math.fsum`
   精确均值去除恒定偏置——相邻块均值之差对整体加常数数学不变，去偏置可避免
   大基线信号在三项相减时灾难性抵消），每个差分用
   `d_k = (C[k+2m] − 2·C[k+m] + C[k]) / m` 得到；平方和用 `math.fsum`
   累加。归一化用无偏因子 `1/(2(count−1))`。
2. **时间戳模式**：只对**有效样本**估计标称周期与窗内均值；均匀完整且窗长为
   整数倍周期时走与定频模式相同的累积和**快路径**（两路径结果一致），缺测/
   不规则时把样本吸附到标称整数网格，用三个单调指针线性扫描三条窗边界。
   无效样本不移动网格、不插值、不补零；半窗覆盖阈值固定为 1/2。
3. **拟合点筛选**：差分样本数 `>= 20` 且方差严格为正。常量信号（方差精确
   为 0，如全 0 或全 3.5）在 log 域无定义，不参与拟合。
4. **分段直线**：点按窗长去重升序后，枚举全部**连续**切分（段间无重叠、无
   缺口），段数 1..3，每段至少 3 点；段内为闭式 OLS，全局用**动态规划**
   `dp[seg][b] = min_a dp[seg−1][a−1] + SSE(a,b)` 求最小原始 SSE，再以
   `SSE + (p−1)·SEGMENT_PENALTY`（罚项常量 `SEGMENT_PENALTY = 1.0`）
   选择段数。并列时取段数更少者。两种模式共用同一拟合器。
5. **搜索预算 vs 无解**：DP 是对全部合法切分的精确穷举，预算常量
   （`_SEARCH_BUDGET`，8·K² 量级）取公开输入边界理论求值上界之上，因此对
   合法输入**不可达**；若预算耗尽抛 `SearchBudgetExhaustedError`，与已证明
   不存在合法切分的 `FitInfeasibleError` 严格区分。
6. **无部分变更**：`analyze` 为纯函数，不改写输入、无全局状态；任何错误都
   先于返回抛出。`FitInfeasibleError.points` 附带已算好的逐窗统计（异常即
   结果的一部分，不属于部分写入）。

## 错误语义

- `AllanTypeError`（同时是 `TypeError`）：类型非法，含对整数参数传 `bool`、
  `valid` 标志不是 `bool`/`0/1`、时间戳或窗长序列类型错误等。
- `AllanValueError`（同时是 `ValueError`）：`NaN/Infinity`、`dt<=0`、
  空序列、长度超 8192、`m<1`、窗长 `<=0`、**时间戳不严格递增或与 samples
  不等长**、`valid` 与 samples 不等长、**混用两种模式的参数**、超大整数
  无法表示为有限 float 或计算溢出等；错误信息定位到参数名与下标
  （如 `timestamps[10]: ...`、`cluster_durations[3]: ...`），不静默纠正、
  不返回 `inf/NaN` 结果。
- `FitInfeasibleError`：合格拟合点不足 3 个（常量零方差、短数据、整段缺测
  导致有效窗对不足等），信息说明原因（证明无解）。
- `SearchBudgetExhaustedError`：搜索预算耗尽（公开边界内不会发生）。
- 以上均继承 `AllanError`。

## 浮点误差口径（不以宽容差掩盖边界）

- 全部计算为 IEEE-754 binary64（Python float）。
- 定频模式测试中核心与**朴素双重循环窗口求和参考**（不去偏置、直接累加）的
  方差相对误差要求 `< 1e-12`（数据量级 O(1)、N≤300）；这是具体测得的数值
  口径，不是对任意量级输入的通用保证——量级接近 `1e308` 或噪声相对基线接近
  `2^-53` 的输入即使去偏置也无法恢复被 binary64 抹掉的低位。
- 时间戳模式的 `count/expected_count/missing_count` 为**精确整数**，由固定
  端点规则枚举确定；测试用**完全独立的按定义枚举参考**（中位数、吸附、窗
  归属、覆盖判定各自重写）逐窗长核对，要求计数完全一致、方差相对误差
  `< 1e-12`；均匀完整数据额外与定频模式互证。手算小例（线性序列、定向
  挖洞、半窗阈值、分数窗长、整段缺测）锁定边界。
- 时间戳网格吸附容差为相对 `1e-9`：只用于吸收均匀时间戳经 binary64 运算后
  约 eps 量级的网格偏差；它不是“接近就合并样本”的钳制，落在格子中间
  （如真实的 1.5 倍周期抖动）的样本不会被错配到相邻格。
- log-log 拟合目标值与**枚举全部切分的独立参考**要求绝对差
  `< 1e-9·max(1,|参考值|)`；断线下标要求完全一致（构造数据无并列）。
- `variance` 精确为 0 的判定来自浮点精确抵消；非二进制可精确表示的常量
  （如重复的 `0.1`）理论上可能留下 `~1e-16` 相对量级残差，这是 binary64
  表示本身的限制，模块不做"接近 0 就当 0"的隐式钳制。

## 已知限制

- 只处理单轴、离线序列；不做在线/流式估计、不做多轴相关、不做去趋势、
  不加窗、不做置信区间或噪声系数换算。
- 时间戳模式不做重采样/插值：缺测样本的位置空着，覆盖不足的窗对直接记
  缺失；标称周期用间隔**中位数**刻画，适合以单个标称速率为主、偶发抖动/
  丢点的记录，不面向连续变速采样。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差与缺失窗被剔除而非替换。
- `cluster_sizes`/`cluster_durations` 与样本均达 8192 时计算量为 O(N·K)
  （约 6.7×10⁷ 次差分级运算），属接口上界规模；均匀完整数据走累积和快路径，
  实测与定频模式同量级（本机约 11 秒）；缺测/不规则数据走线性扫描路径，
  同规模约 30 秒。短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（两种时间轴模式的核心算法与全部数据结构/
  错误类型）。
- `tests/test_allan.py`：定频模式必要测试（朴素窗口求和、itertools 枚举
  分段 + 独立 OLS 参考，期望值不来自核心）。
- `tests/test_timestamped.py`：时间戳模式必要测试（按定义独立枚举的参考、
  手算小例、与定频模式互证、不规则采样、整段缺测、非法输入）。
- `demo.py`：固定种子（20260922/20260924）的两段确定性演示：定频
  "白噪声 + 恒定漂移"分解，以及时间戳模式下的不规则采样 + 中段整段缺测；
  打印正常结果、缺失计数、拟合（或无解），并真实触发 NaN/bool/非递增
  时间戳/标志长度不符/混用模式/非法窗长等拒绝边界。
