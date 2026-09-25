# 固定速率陀螺仪 Allan 误差分解（离线批量 + 在线累积）

对定频采样的单轴角速率记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。

两种使用方式：

- `analyze(samples, dt, cluster_sizes)`：离线批量入口，单次最多 8192 个样本；
- `OnlineAllan(dt, cluster_sizes)`：**在线累积器**，`feed` 任意长度 chunk、
  随时 `query` 当前完整前缀的各尺度 Allan 方差、有效差分数与分段拟合；
  只保留最大尺度所需的滑动样本窗口与每尺度 O(1) 累积量，样本数可远超
  批量上限，内存不随历史长度增长。

> 拟合得到的斜率只是对**本批数据** log-log 形态的经验描述，不命名、不暗示
> 任何临床结论或硬件指标保证，本模块也不宣称完整兼容任何行业标准；它只实现
> 本题契约内的数学子集。

## Python 版本与依赖

- Python **3.14.7**（仅使用标准库：`math`、`dataclasses`、`collections.abc`）。
- 安装 Python 后**无需安装任何第三方依赖**，无需网络/账号/硬件/数据库/Docker。

## 运行命令

```bash
# 必要测试（独立参考，不调用被测核心产生期望值）
python -m unittest discover -s tests -v

# 固定输入演示（确定性合成数据；实测约 0.5 秒内完成，远小于 90 秒预算）
python demo.py
```

## 在线累积器（可复现使用路径）

```python
from allan_deviation import OnlineAllan

acc = OnlineAllan(0.01, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
acc.feed(chunk_1)          # 任意长度、任意次数；单点、素数块、整块均可
acc.feed(chunk_2)
points = acc.query_points()   # 当前前缀逐 m 的 AllanPoint（不拟合、不改状态）
result = acc.query()          # 与 analyze(当前完整前缀) 相同形状的 AllanResult
```

- `feed(chunk)` 返回消费的样本数；**先整块校验再写入**，任一元素非法
  （非数值、`bool`、`NaN`/`±Infinity`、不可表示为有限 float 的超大 int）
  则整块拒绝并抛 `AllanTypeError`/`AllanValueError`（信息定位到
  `chunk[i]`），已累积状态完全不变，不会吃掉前几个合法元素；空 chunk 是
  无副作用的 no-op。
- `dt` 与 `cluster_sizes` 仅在构造时校验并固定，**创建后不可更改**
  （只读属性，赋值抛 `AttributeError`；尺度以 tuple 副本保存）。
- `query_points()` 随时返回逐 m 统计（`variance`/`adev`/`count`/
  `eligible_for_fit`），不做拟合、永不抛 `FitInfeasibleError`；
  `query()` 额外做分段拟合，合格点不足时与批量入口一样抛
  `FitInfeasibleError`（`.points` 附带当前统计）。两者都是只读操作，
  **分段拟合不改变任何累积状态**，可任意次数调用。
- 尺度 m 的有效差分数恒为 `max(0, N - 2m + 1)`：不足 2m 的尾部**不补零**，
  与 `analyze` 对同一前缀的结果一致（数值上为不同求和次序的浮点实现，
  相对差实测 `< 1e-12`，见下节误差口径）。
- 内存只保留 `min(2*max(m), 已见样本数)` 个滑动样本（`retained_samples`
  可直接查看）与每尺度 O(1) 累积量，**不缓存完整历史**；样本数可远超
  `MAX_SAMPLES`（测试覆盖 22 万样本，堆占用零增长）。
- 数值稳定方案：进入累积前减去首个非空 chunk 的精确均值（差分对整体常数
  平移数学不变，去偏置避免大直流基线下的灾难性抵消）；滑动窗口和与差分
  平方和均用 **Neumaier 补偿累加**。
- 明确不做：缺测填充、传感器在线连接、多轴、去趋势、置信区间。

### 失败示例（demo.py 现场触发）

```text
acc.feed([1.0, 2.0, float("nan"), 4.0])
# -> AllanValueError: chunk[2]: value must be finite (NaN/Infinity rejected)
#    拒绝后 n_samples 不变（前 2 项未被吃掉），可继续正常 feed
acc.dt = 0.5
# -> AttributeError（创建后 dt 不可改；cluster_sizes 同理）
```

## 批量接口

```python
from allan_deviation import analyze
result = analyze(samples, dt, cluster_sizes)
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
   结果的一部分，不属于部分写入）。`OnlineAllan.feed` 先整块校验再写入，
   非法 chunk 不留下任何部分消费。
6. **在线累积**：每个尺度 m 维护"最近 m 个残差之和""之前 m 个残差之和"
   （均为 Neumaier 补偿和）与差分平方的补偿累积和；每个新样本以 O(1) 滑窗
   更新各尺度，新差分 `d = (后 m 和 − 前 m 和)/m` 的平方即时累进。滑动窗口
   用惰性增长的环形缓冲（上界 `2*max(m)`），读取最旧样本先于覆盖写入。
   残差 = 样本 − 首个非空 chunk 的精确均值（`math.fsum`），与批量入口的
   去偏置同理，差分结果数学不变。

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
- 在线累积器与批量入口是同一数学量的不同浮点求和次序（滑窗补偿和 vs
  前缀和）：测试中 `OnlineAllan` 与**朴素双重循环参考**在单点喂入的每个
  前缀、素数块喂入的每个块边界上相对差要求 `< 1e-12`（O(1) 数据），6 万
  样本长流末端要求 `< 1e-10`；与 `analyze` 同前缀结果相对差 `< 1e-12`，
  分段断点要求完全一致。大直流偏置（1e9 基线）按表示极限分档断言
  （`< 1e-7`，与批量测试同口径）。

## 已知限制

- 只处理单轴、定频序列；不做多轴相关、去趋势、加窗、置信区间或噪声系数
  换算；**不做缺测填充**（feed 的每个元素都必须是有限数值），**不接传感器
  在线连接**（累积器只接受调用方传入的数值 chunk）。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时批量计算量为 O(N·K)（约 6.7×10⁷
  次差分级运算），属接口上界规模；短演示不穷举该规模。在线累积器每样本
  每尺度 O(1)，内存 O(2·max(m) + K)，但 max(m) 本身巨大时滑动窗口
  （2·max(m) 个 float）相应占内存，这是算法的固有需求。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（批量 `analyze`、在线 `OnlineAllan`、
  全部数据结构/错误类型）。
- `tests/test_allan.py`：批量入口必要测试；参考实现（朴素窗口求和、
  itertools 枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心。
- `tests/test_online.py`：在线累积器必要测试；同一合成信号按单点、素数
  块、整块送入，在每个前缀与独立双循环公式比较；覆盖常量信号、大直流
  偏置叠小变化、刚够 2m 个点的边界、超批量上限长流、内存不随历史增长、
  非法 chunk 整块拒绝、dt/尺度不可变、query 不改变累积状态。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，打印批量分解
  结果、经验区间、理论交叉点对照；在线部分演示素数块长流喂入（4 万样本，
  超批量上限）、与批量入口的同前缀一致性、独立双循环抽查，并真实触发
  NaN 样本、bool 类型、非法 chunk 整块拒绝、dt 不可改与短数据无解边界。
