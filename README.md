# 固定速率陀螺仪 Allan 误差分解（离线批量 + 在线累积）

对定频采样的单轴角速率记录，计算**重叠 Allan 方差**（overlapping Allan
variance），并在 log(聚类时间)–log(Allan 标准差) 平面上做**连续分段直线**
分解，用于经验性观察白噪声区、漂移区等形态。提供两个入口：

- `analyze(samples, dt, cluster_sizes)`：离线批量入口（旧接口，保持不变）；
- `AllanAccumulator(dt, cluster_sizes)`：**在线累积器**，`feed(chunk)` 接受
  任意长度分块的长流输入，随时 `query()` 当前完整前缀的同口径结果，
  内存只占 O(2·max(m) + K)，不缓存完整历史。

> 拟合得到的斜率只是对**本批数据** log-log 形态的经验描述，不命名、不暗示
> 任何临床结论或硬件指标保证，本模块也不宣称完整兼容任何行业标准；它只实现
> 本题契约内的数学子集。

## Python 版本与依赖

- Python **3.14.7**（仅使用标准库：`math`、`array`、`dataclasses`、
  `collections.abc`）。
- 安装 Python 后**无需安装任何第三方依赖**，无需网络/账号/硬件/数据库/Docker。

## 运行命令

```bash
# 必要测试（独立参考，不调用被测核心产生期望值）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示（确定性合成数据；实测约 5 秒内完成，远小于 90 秒预算）
./.venv/bin/python demo.py
```

## 公开接口

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

## 在线累积器（长流输入）

```python
from allan_deviation import AllanAccumulator

acc = AllanAccumulator(dt, cluster_sizes)   # 创建时校验并冻结 dt 与尺度
acc.feed(chunk)                              # 任意长度分块，可反复调用
result = acc.query()                         # 当前完整前缀的 AllanResult
```

- `dt`、`cluster_sizes`：校验规则与批量入口相同；创建后**不可修改**
  （只读属性，赋值抛 `AttributeError`；`cluster_sizes` 返回 tuple 副本，
  之后改动传入的 list 不影响累积器）。
- `feed(chunk)`：`chunk` 为任意长度的有限实数序列（空块是合法无操作）。
  **先整块校验、后应用**：任一元素非法（非数值、bool、NaN/±Infinity）都抛
  `AllanTypeError`/`AllanValueError`（信息定位到 `chunk[i]`），且块内
  **任何样本都不会被计入**——失败不留下部分变更。返回累计样本数。
  样本总数不受 `MAX_SAMPLES` 限制，可远超单次批量上限 8192。
- `query()`：返回与 `analyze` 作用于当前完整前缀**相同口径**的
  `AllanResult`（逐 m 方差、有效差分数 `count`、分段拟合）。尾部不足
  2m 个样本**不补零**，只体现为该尺度的 `count` 更小。`query` 只读
  累积状态、可任意次调用、可与 `feed` 任意交替；可拟合点不足时与批量
  入口一样抛 `FitInfeasibleError`（`.points` 附带逐 m 统计）。
- 只读属性：`dt`、`cluster_sizes`、`n_samples`。

可复现使用路径（确定性合成数据，标准库即可运行）：

```bash
./.venv/bin/python - <<'EOF'
import random
from allan_deviation import AllanAccumulator, analyze

rng = random.Random(20260922)
samples = [rng.gauss(0.0, 1.0) + 0.01 * i for i in range(2048)]
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

acc = AllanAccumulator(0.01, sizes)
for i in range(0, len(samples), 97):        # 任意分块（97 一块）
    acc.feed(samples[i:i + 97])
online = acc.query()
batch = analyze(samples, 0.01, sizes)        # 旧批量入口保留
worst = max(abs(a.variance - b.variance) / b.variance
            for a, b in zip(online.points, batch.points))
print(f"在线 vs 批量最大相对差: {worst:.3g}")   # 实测 ~1e-14
print(f"段数: {online.fit.segment_count}, 断点: "
      f"{[(bp.m, bp.tau) for bp in online.fit.breakpoints]}")
EOF
```

失败示例（实际触发，非预录）：

```python
acc.feed([1.0, 2.0, float("nan")])
# AllanValueError: chunk[2]: value must be finite (NaN/Infinity rejected)
# —— 整块拒绝：n_samples 不变，块内前两个合法样本也不会被计入。
acc.dt = 0.1
# AttributeError: property 'dt' of 'AllanAccumulator' object has no setter
```

`./.venv/bin/python demo.py` 的在线段落会现场演示以上两条路径，并用
tracemalloc 展示历史长度加倍（6 万 → 12 万样本）后累积器内存不再增长。

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
6. **在线累积**：`AllanAccumulator` 对每个新样本以 O(K) 更新——环形缓冲
   保存最近 `2*max(m)` 个**去基准**样本（基准 = 首个样本；相邻块均值之差
   对整体加常数数学不变，与离线均值去偏置同效），每个尺度维护"最近 m 个
   之和 / 其前 m 个之和"两个滑动和与差分平方和，新样本到达时若凑满 2m
   个则产生一个新差分 `d_k` 并累加 `d_k²`。所有滑动和与平方和均用
   **Neumaier 补偿累加**；内存 O(2·max(m) + K)，与历史长度无关。
   `query` 只做 `variance = Σd_k²/(2(count−1))` 与既有的分段拟合，不改动
   任何累积量；分块方式不影响结果（同一序列按单点、素数块、整块送入，
   累积状态逐位相同）。

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
- 在线累积器与批量入口的数学定义相同，但浮点求值次序不同（去基准滑动和
  vs 均值去偏置前缀和），二者不保证逐位一致；测试中在线 `query` 与同一
  前缀的朴素双循环参考在 O(1) 数据（N≤300）上相对差 `< 1e-12`，与批量
  `analyze` 的逐 m 方差相对差亦 `< 1e-12`（实测约 1e-14）；大直流偏置
  分档同批量口径（1e6 基线 `< 1e-9`，1e9 基线 `< 1e-7`）。同一序列不同
  分块送法则逐位一致。

## 已知限制

- 只处理单轴、定频序列；不做多轴相关、去趋势、加窗、置信区间或噪声系数
  换算；**不做缺测填充**——在线入口假定样本按固定 dt 连续到达，断档需要
  调用方自行重建累积器；不附带任何传感器在线连接。
- 在线累积器的尺度集合在创建时固定，之后不能增删 m（需新建实例）；
  其内存为 O(2·max(m) + K)，`max(m)` 很大时环形缓冲相应占
  `16·max(m)` 字节，这是"只保留最大尺度所需滑动样本"的固有下界。
- Allan 方差之外的变体（修正 Allan、Hadamard、时间序列等）未实现。
- 分段拟合在 log 域进行；零/负方差点被剔除而非替换。
- `cluster_sizes` 达 8192 且样本达 8192 时批量计算量为 O(N·K)（约
  6.7×10⁷ 次差分级运算），属接口上界规模；在线入口为每样本 O(K)，
  短演示不穷举该规模。
- 斜率命名保持中性（"小段/大段斜率"），不映射为临床或硬件保证术语。

## 文件

- `allan_deviation.py`：本题模块（核心算法、在线累积器与全部数据结构/
  错误类型）。
- `tests/test_allan.py`：批量入口的必要测试；参考实现（朴素窗口求和、
  itertools 枚举分段 + 独立 OLS）与被测核心完全独立，期望值不来自核心。
- `tests/test_online.py`：在线累积器的必要测试；同一合成信号按单点、
  素数块、整块送入并在每个前缀对照朴素双循环参考，覆盖常量、大直流
  偏置、2m 边界、非法 chunk 整块拒绝、dt/尺度冻结、query 无状态副作用
  与内存不随历史增长。
- `demo.py`：固定种子（20260922）合成"白噪声 + 恒定漂移"，打印离线分解
  结果、经验区间、理论交叉点对照，并演示在线分块送样、中途查询、长流
  内存不变，真实触发 NaN、bool 类型拒绝、短数据无解与非法 chunk 边界。
