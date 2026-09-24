# 固定速率陀螺仪 Allan 误差分解（离线可复用模块）

对单轴角速率离线记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。支持两种记录形态：

- **定频完整记录**：直接给样本序列与采样周期 `dt`（旧能力，行为不变）；
- **不规则采样 / 含缺测记录**：通过仅关键字参数 `timestamps`（严格递增、
  不要求均匀）与 `valid`（逐样本有效标志）给入，统计按**实际时间窗口**内的
  有效样本均值计算；任一窗覆盖不足的相邻窗对记为**缺失**。

> 拟合得到的斜率只是对**本批离线数据** log-log 形态的经验描述，不命名、不暗示
> 任何临床结论或硬件指标保证，本模块也不宣称完整兼容任何行业标准；它只实现
> 本题契约内的数学子集。

## Python 版本与依赖

- Python **3.14.7**（仅使用标准库：`math`、`dataclasses`、`collections.abc`）。
- 安装 Python 后**无需安装任何第三方依赖**，无需网络/账号/硬件/数据库/Docker。

## 运行命令

```bash
# 全部必要测试（独立参考，不调用被测核心产生期望值）
python -m unittest discover -s tests -v

# 固定输入演示（确定性合成数据；定频 + 不规则采样/缺测两段，实测约 0.1 秒
# 内完成，远小于 90 秒预算；无 sleep、无预录结果）
python demo.py
```

## 公开接口

```python
from allan_deviation import analyze

# 定频完整记录（旧签名，行为不变）
result = analyze(samples, dt, cluster_sizes)

# 不规则采样 / 缺测记录（仅关键字的新选项，可单独或组合使用）
result = analyze(samples, dt, cluster_sizes,
                 timestamps=ts,          # 与 samples 等长、严格递增的秒级时间戳
                 valid=flags,            # 与 samples 等长的真正 bool 序列
                 min_window_samples=q)   # 每窗最少有效样本数，默认等于 m
```

- `samples`：单轴角速率序列，长度 **1..8192**（接口上限），元素为
  `int`/`float` 且全部有限；`bool`、`str`、`complex` 等按类型错误拒绝，
  `NaN`/`±Infinity` 按取值错误拒绝。
- `dt`：**名义**采样周期（秒），`int`/`float`、有限且 `> 0`；`bool` 拒绝。
  定频路径下即实际采样周期；窗口路径下同时是网格步长与聚类时间单位
  （`tau = m*dt`，窗口长度 `m*dt` 秒）。
- `cluster_sizes`：聚类大小 `m`（以 `dt` 为单位的窗口长度）的有限序列
  （list/tuple/range），长度 **1..8192**；每个元素必须是**真正的 `int`**
  （`bool`、`1.0` 拒绝）且 `m >= 1`，且 `m*dt` 必须可表示为有限 float。
  允许**重复**与**非单调**：输出点顺序与输入一致；拟合前自动按 `m` 去重并
  升序排列。
- `timestamps`（可选，仅关键字）：与 `samples` 等长的严格递增有限实数序列
  （秒）。不要求均匀间隔；相等或倒退的时间戳按取值错误拒绝。不给定时走
  定频路径。
- `valid`（可选，仅关键字）：与 `samples` 等长的**真正 `bool`** 序列
  （`0/1` 等 `int` 不隐式接受）；`False` 表示该样本缺测，窗内统计时跳过。
  可单独用于定频记录的缺测，也可与 `timestamps` 组合。
- `min_window_samples`（可选，仅关键字）：真正的 `int` 且 `>= 1`。一个时间
  窗口内有效样本数少于该值时，包含该窗的相邻窗对记为**缺失**。默认 `m`
  （即每个 `m*dt` 窗至少要有 `m` 个有效样本才算充分覆盖）。

返回 `AllanResult`（不可变 dataclass）：

| 字段 | 含义 |
|---|---|
| `points[i]` | 与输入 `cluster_sizes` 一一对应的 `AllanPoint(m, tau, variance, adev, count, eligible_for_fit, positions, missing)` |
| `fit` | `FitResult`：`segments`、`breakpoints`、`slopes`、`intercepts`、`sse`、`rmse`、`residuals`、`fitted_ms`、`segment_count`、`objective` |
| `dt` / `n_samples` | 回显校验后的 dt（转为 float）与样本数 |

`AllanPoint`：

- `tau = m * dt`；
- `positions`：网格上枚举的相邻窗对位置总数。定频路径恒为
  `max(0, N − 2m + 1)`；窗口路径为
  `max(0, floor((t_last − t0)/dt) − 2m + 2)`（端点取整带 1e-9 相对容差，
  见"窗口端点规则"）；
- `missing`：因某一窗有效样本数 `< min_window_samples` 而跳过的窗对数
  （定频完整输入恒为 0）；
- `count = positions − missing`：**实际**进入统计的有效相邻均值差样本数；
- `variance = Σ d_k² / (2·(count−1))`，`d_k = 有效窗2均值 − 有效窗1均值`，
  `count < 2` 时为 `None`（无法做无偏估计）；
- `adev = sqrt(variance)`；
- 仅 `count >= 20` 且 `variance > 0` 的点 `eligible_for_fit=True`。

`positions`/`missing` 是新增字段，排在原有字段之后且带默认值，旧代码按位置
构造 `AllanPoint(m, tau, var, adev, count, eligible)` 的用法继续可用。

`Segment`：`y = slope·x + intercept`（`x=log(tau)`，`y=log(adev)`），并带
`m_start/m_end`、`tau_start/tau_end`、`n_points`。`Breakpoint` 定义为右侧段
第一个拟合点（`m`、`tau`）。`residuals` 与 `fitted_ms` 对齐，为
观测 `log(adev)` 减模型值。`objective = sse + (segment_count−1)·SEGMENT_PENALTY`。

## 算法与设计取舍

1. **重叠 Allan 方差（定频路径）**：先对样本做前缀累积和（此前用
   `math.fsum` 精确均值去除恒定偏置——相邻块均值之差对整体加常数数学不变，
   去偏置可避免大基线信号在三项相减时灾难性抵消），每个差分用
   `d_k = (C[k+2m] − 2·C[k+m] + C[k]) / m` 得到；平方和用 `math.fsum`
   累加。归一化用无偏因子 `1/(2(count−1))`（有限记录的标准重叠估计写法）。
2. **实际时间窗口路径**：给入 `timestamps` 和/或 `valid` 时启用。只对有效
   样本构造（时间戳, 去偏置值）的前缀和，用 `bisect_left` 在 O(log N) 内
   定位每个窗口边界、由前缀和差求窗内均值；每个 m 扫描其全部网格位置，
   总复杂度 O(K_positions·log N)。覆盖不足的窗对跳过，不补零、不外推。
   **窗口端点规则（固定）**：网格锚点为首时间戳 `t0`，第 k 对为半开区间
   `[t0+k·dt, t0+(k+m)·dt)` 与 `[t0+(k+m)·dt, t0+(k+2m)·dt)`，恰落在
   边界上的样本归入右侧窗口；枚举到右窗的最后一个网格格点起点
   `t0+(k+2m−1)·dt <= t_last` 为止，故
   `positions = max(0, floor((t_last−t0)/dt) − 2m + 2)`。floor 前带
   1e-9 相对容差吸收 `i*dt` 网格的 binary64 舍入（容差远小于半格 0.5，
   不会吞掉真实的末端半窗）。该规则下**均匀完整输入**（`t_last=(N−1)dt`）
   恰好还原 `positions = N−2m+1`，与定频路径逐对一致。
3. **拟合点筛选**：有效差分样本数 `>= 20` 且方差严格为正。常量信号（方差
   精确为 0，如全 0 或全 3.5）在 log 域无定义，不参与拟合；缺测过多导致
   有效对不足同样不参与。
4. **分段直线**：点按 `m` 去重升序后，枚举全部**连续**切分（段间无重叠、无
   缺口），段数 1..3，每段至少 3 点；段内为闭式 OLS，全局用**动态规划**
   `dp[seg][b] = min_a dp[seg−1][a−1] + SSE(a,b)` 求最小原始 SSE，再以
   `SSE + (p−1)·SEGMENT_PENALTY`（罚项常量 `SEGMENT_PENALTY = 1.0`，
   单位为 log(adev) 残差平方）选择段数。并列时取段数更少者。窗口路径复用
   同一拟合链路，不做任何改动。
5. **搜索预算 vs 无解**：DP 是对全部合法切分的精确穷举，预算常量
   （`_SEARCH_BUDGET`，8·K² 量级）取公开输入边界理论求值上界之上，因此对
   合法输入**不可达**；若预算耗尽抛 `SearchBudgetExhaustedError`（无法证明
   最优），与已证明不存在合法切分的 `FitInfeasibleError` 严格区分。
6. **无部分变更**：`analyze` 为纯函数，不改写输入、无全局状态；新参数全部
   先校验后计算，任何错误都先于返回抛出。`FitInfeasibleError.points` 附带
   已算好的逐 m 统计（含 `positions/missing`，异常即结果的一部分，不属于
   部分写入）。

## 错误语义

- `AllanTypeError`（同时是 `TypeError`）：类型非法，含对整数参数传 `bool`；
  `timestamps`/`valid`/`min_window_samples` 类型错误（如 `valid=[1,0]`）。
- `AllanValueError`（同时是 `ValueError`）：`NaN/Infinity`、`dt<=0`、
  空序列（`samples` 或 `cluster_sizes`）、长度超 8192、`m<1`、超大整数无法
  表示为有限 float 或 `m*dt` 溢出、样本量级使均值/方差计算溢出；
  `timestamps` 与 `samples` 不等长或非严格递增、`valid` 与 `samples` 不等长、
  `min_window_samples < 1`。错误信息定位到参数名与下标
  （如 `timestamps[3]: ...`、`valid[7]: ...`），不静默纠正、不返回
  `inf/NaN` 结果。
- `FitInfeasibleError`：合格拟合点不足 3 个（常量零方差、短数据、覆盖不足、
  整段缺测等），信息说明原因（证明无解）。
- `SearchBudgetExhaustedError`：搜索预算耗尽（公开边界内不会发生）。
- 以上均继承 `AllanError`。

## 浮点误差口径（不以宽容差掩盖边界）

- 全部计算为 IEEE-754 binary64（Python float）。
- 测试中定频核心与**朴素双重循环窗口求和参考**（不去偏置、直接累加）的方差
  相对误差要求 `< 1e-12`（数据量级 O(1)、N≤300）；这是具体测得的数值口径，
  不是对任意量级输入的通用保证——量级接近 `1e308` 或噪声相对基线接近
  `2^-53` 的输入即使去偏置也无法恢复被 binary64 抹掉的低位。
- **窗口路径**与独立参考（按半开区间定义逐对取 `fsum` 均值）的方差相对误差
  要求 `< 1e-11`；均匀完整输入走窗口路径与定频路径结果的相对差 `< 1e-12`
  （两条路径的浮点分组不同——窗口路径是"两个均值分别相除再相减"，定频路径
  是"前缀和三项组合后再除 m"——差异为 1 ULP 量级，属预期）。
- log-log 拟合目标值与**枚举全部切分的独立参考**要求绝对差
  `< 1e-9·max(1,|参考值|)`；断线下标要求完全一致（构造数据无并列）。
- `variance` 精确为 0 的判定来自浮点精确抵消；非二进制可精确表示的常量
  （如重复的 `0.1`）理论上可能留下 `~1e-16` 相对量级残差，这是 binary64
  表示本身的限制，模块不做"接近 0 就当 0"的隐式钳制。

## 已知限制

- 只处理单轴、离线记录；不做在线/流式估计、不做多轴相关、不做去趋势、
  不加窗、不做置信区间或噪声系数换算。
- 窗口网格以**首时间戳为锚、名义 dt 为固定步长**；不做从抖动时间戳反演
  相位/频率，也不对不规则间隔做重采样插值——缺测窗口直接记缺失。
- 覆盖判据只有"窗内有效样本数下限"一种；不按时间占比或加权均值处理，
  有效样本在窗内的具体位置不影响均值（仍是窗内样本的算术平均）。
- 有效差分不足（`count<20`）的 m 不参与拟合；整段缺测时所有 m 均无有效点，
  结果以 `FitInfeasibleError.points` 附带，而不是返回空拟合。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时，定频路径计算量为 O(N·K)
  （约 6.7×10⁷ 次级运算），窗口路径为 O(K_positions·log N)，属接口上界
  规模；短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（定频 + 时间窗口两条计算路径，全部数据
  结构/错误类型）。
- `tests/test_allan.py`：定频路径必要测试；参考实现（朴素窗口求和、itertools
  枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心。
- `tests/test_windowed.py`：不规则采样 / 缺测路径必要测试；独立参考
  `naive_windowed` 按半开区间定义逐对直算，覆盖正常、端点边界、整段缺测、
  覆盖阈值与全部新增非法输入。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，A 段演示定频
  完整记录，B 段演示同信号的不规则采样（间隔 ±35% 抖动）与整段缺测，打印
  count/positions/missing 与分解结果，并真实触发 NaN、bool、非递增时间戳、
  标志类型错误及短数据/整段缺测无解边界。
