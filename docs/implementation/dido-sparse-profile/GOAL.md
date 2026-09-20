# 可在本 worktree 新会话中直接执行的 goal 提示词

下面整段是未来执行任务的提示词；保存此文件不代表已启动实验。

---

请将以下任务作为本会话的 goal，分阶段持续推进。先核实代码、论文和数据，再实现、测试、部署和做有界实验；不要停在概念建议，也不要为达到速度目标伪造或隐藏负结果。遇到需要额外权限、缺失数据或新增训练的问题，明确报告并请求方向。

## 1. 目标与授权边界

- 在 DreamWAM 现有 hybrid sparse/reuse 实现上，研究 action–video 的真实依赖，并借鉴 DIDO 对交互区域和动态信息的分析，回答“哪些视觉 token 值得保留、哪些应该重算、何时刷新”。
- 目标是完整动作预测相对同输入、同后端的强化 Dense 达到 **>1.5× 推理加速**，同时以真实 LIBERO 闭环评估控制能力，而不是只报告 Attention kernel 加速或动作向量相似度。
- 优先在同等 SR/动作偏差下减少保留量；如果更低比例不再改善时延或明显破坏控制，就保留实测 Pareto 候选，不无限下降比例。
- 本阶段默认不训练、不蒸馏、不修改原始 checkpoint。DIDO 的新增 token、框监督、表征对齐与蒸馏是训练方法，不能当作现有模型无需训练即可获得的能力。需要训练或引入重型检测器时先说明必要性、资源和权限。
- 只使用 H100 已授权资源；先离线、小规模筛选，再进行有界配对闭环。启动昂贵阶段前说明 GPU、候选数、调用/episode 上限和预估耗时。不自动启动 50 对、500 episode 或历史队列。

## 2. 严格使用已建好的迭代链路

- 当前工作目录应为 `/home/winbeau/Papers/ICLR2027-WAM-SA/.trees/dido-sparse-profile`，分支 `experiment/dido-sparse-profile`。不要切回 main，不改别的工作树。
- H100 对应目录为 `/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile`。先读本目录 `AGENTS.md` 与 `docs/implementation/dido-sparse-profile/WORKFLOW.md`。
- 每个逻辑阶段及时 `git add <明确文件>`、`git commit`、`git push`。使用 `bash deployment/h100/sync-dido-worktree.sh` 推送并在 H100 执行干净工作树上的 `pull --ff-only`，核对完全一致的 commit。
- 代码只在本地编辑；测试、模型运行和评测只在 H100。不得直接改服务器代码、复制未提交代码绕过 Git、覆盖其他人的更改、升级依赖或重建已验证环境。
- 运行中的实验使用独立、固定 commit 的服务器 worktree；不得边跑边 pull。记录代码、配置、checkpoint、输入 manifest 和输出 artifact 的 hash。
- 若确实需要改 action-eval 的 YAML 或代码，为该仓库另开同形式的 worktree/分支并建立相同链路；先读其配置和适配器技能。不要把选择算法塞进 evaluator。

## 3. 阅读论文，并明确我们借鉴的是什么

- 精读 https://arxiv.org/pdf/2609.15570v2 ，重点核对 §3.3、Fig.3、Appendix A、Appendix D.2；追踪 https://github.com/LoveJu1y/DIDO-WAM/ 和 https://loveju1y.github.io/DIDO/ 的实际发布状态。
- 论文的动态 refinement 使用当前/未来帧的 V 特征差，保留高动态区域细粒度 token，并池化低动态区域 K/V；这与直接删除低分 token 不同。移植时先核实 DreamWAM 的真实空间网格、相机拼接、时间帧和 RoPE，不能照抄论文网格或 token 数。
- 论文另有 action V-attribution 分析。先核实它的具体定义与实现；如果作者没有公开，自己的 Attention×Value 范数或干预分数必须标记为我们定义的代理，不能宣称精确复现。
- 分开记录“论文报告”“代码已实现”“本实验验证”“仍是假设”。作者模型的速度/SR/交互 token 结论不能直接归因给 DreamWAM。

## 4. 先弄清 raw data，而不是自行替换概念

- 明确列出：论文原始机器人 demonstrations、自动生成的对象/夹爪轨迹或框、作者 attribution/图表原始记录，以及本模型运行时 Q/K/V/latent 原始张量。这些不是同一种数据。
- 截至本交接建立时，作者代码仓库仅 README，说明代码待发布；尚未核实作者 raw data 下载入口。重新核实公开链接、版本、许可、格式与数据覆盖，不能声称已获得未发布数据。
- 如果用户指作者私有的逐样本记录且公开渠道没有，询问具体路径/链接；可以继续做不依赖它的代码审计和本地模型 profiling，但不能把这部分报为完成。
- 如改用公开 LIBERO demonstrations 或已有闭环观测，明确标注为替代数据/我们自采数据，说明差异；不要将论文图片截图、聚合数字或新采样本伪装成论文 raw data。
- 数据转换必须可重现、可校验，并冻结 development 与 confirmation 的 episode/task 划分。未来图像、真值动作、对象框、模拟器特权状态可作离线分析参考，不得流入在线推理输入。任何学习到的先验仅由 development 数据产生，再冻结到 confirmation。

## 5. 先继承并复查现有结果，不从零重写

- 阅读 `docs/action-eval/decision-support-method-20260920.md`、`hybrid-routing-results-20260920.md`、`hybrid-routing-evidence-20260920.json` 和 `dreamwam/sparse/hybrid/`。
- 已有约 2.08× 热态推理候选是 D0 + R1–9 的**视觉特征复用**，实际读取 56/294 个视觉 token，按帧 `[19,19,18]` 均匀选择。不是 action-aware，不是纯结构复用；配置中的 Q=10% 在无 Sparse 步时不生效。
- 历史 Dense/候选各 3/3 只构成开发 pilot，episode 耗时并未改善；不能当作 SR 保持证明，也不能把重复 Dense 当独立样本。
- 当前 A→V + 一跳 V→V 未稳定优于均匀选择，纯结构复用只测到约 1.06–1.13×；自适应 refresh 尚未实现。保留这些负结果及它们的实验条件。
- 外部输入只有当前双相机图像、本体状态、指令；32 个 action token 从噪声开始去噪，不是真值动作。当前共有 294 个视觉 token，保留所有 action token 和原来的 10 步动作去噪。

## 6. 建立真正的 action–video sparse profile

- 沿 Dense 去噪轨迹按 step/layer/head/frame/camera/空间位置记录 A→V、V→V、V 范数、当前/未来帧的 V 动态，以及相邻去噪步的变化；保存可重放 raw arrays 和元数据，不只导出热力图。
- 先做有界多层、多步试点，明确探测层/头和采样策略；不要只看第一层首步的噪声 action query，也不要未经预算就保存全部巨型 Attention 矩阵。
- 分清“视频时间上的动态”与“去噪时间上的漂移”。可研究 value-aware A→V 贡献，但 Attention 权重、V 范数、Attention×Value 都先视为代理。
- 保留原始 joint Softmax 的全部 action/visual key 分母、mask 与原始位置。AV 表示 action query 读取 visual key，不要混淆方向。
- 同时测量对完整动作序列、实际执行的前 10 步、夹爪符号的影响，以及未来 latent/视频侧误差；不要用生成误差直接替代控制价值。

## 7. 用 raw data 检验“该留谁”

- 若有合法可用的对象/夹爪框、轨迹或分割，在离线把它们映射到真实 token 网格，区分夹爪、目标物、接触邻域、动态区域与背景；记录标注误差和未知区域。
- 比较均匀/随机、A→V、value-aware A→V、V 动态、A→V+V→V、融合信号在相同预算下的覆盖率、动作敏感性和保留效率；加入视觉侧代理对照。
- 进行等预算删除、替换、重算等干预，区分直接 AV 读取与 VV 上下文支撑。干预是诊断，不等于可加速执行。
- 对高/低动态区域做分层分析，检验早步信号是否足够、后期交互区域是否才变得重要。不能先假定 DIDO 的规律在本模型成立。
- 原始数据与打分表写入 manifest；即使某种语义标注不可用，也要清楚指出实际跑了哪些分析，不能把 V 动态代理称为真实对象/接触标签。

## 8. 实现可消融、低耦合的选择与执行

- 独立模块负责 profile、评分、预算分配、集合/区域选择、压缩/打包、缓存、调度与报告。在线 runtime 不依赖离线教师或未来真值。
- 独立控制读取集合 R 与重算集合 U。特征复用模式维护 `R_new - R_old ⊆ U ⊆ R_new`；精确报告实际 token 数、探测开销及更新年龄，不能为补上下文偷偷扩大预算。
- 把 DIDO 启发的“高动态细粒度保留 + 低动态摘要池化”作为单独实验分支，与 hard top-k deletion 公平比较；设计清楚 pooled K/V 的位置、mask、token multiplicity、相机边界及混合新旧状态语义。先验证数值与可见性，再做速度实验。
- 保留独立开关：动作信号、V 动态、VV 支撑、背景摘要、帧配额、Q/KV 预算、刷新策略、features/structure 两类复用、eager/CUDA graph。
- 选择规则必须给出公式、阈值/预算来源与实际索引示例；任何组合权重需由 development 消融支持，不能只凭直觉设定。

## 9. 用实验决定在哪一步 Sparse，而不是手选

- 使用或扩展现有 explicit/periodic/frozen-profile 日程接口，比较无刷新、关键位置单次刷新和少量多次刷新；用固定上限控制候选数。
- 分开实验“刷新稀疏结构”与“重算视觉特征”。不能用 features 模式的速度宣称每步 fresh Q/K/V 也达到同样加速。
- 若实现 adaptive refresh，先在 development 上校准可在线计算的漂移指标和阈值，再冻结；计入监测成本，并报告触发次数/位置/原因。
- 为弱相关、异常值、预算失效设计可审计 fallback；fallback 的频率、额外计算与延迟必须计入结果。

## 10. 测试与性能计量

- 在 H100 运行 CPU 单元测试、真实 CUDA eager/graph 一致性、真实 checkpoint adapter 验证。覆盖满预算退化、集合约束、mask/RoPE、frame/camera 映射、重复调用、更换观测、reset、跨请求隔离和旧缓存污染。
- 强化 Dense、现有 uniform feature-reuse 和新方法在相同输入、checkpoint、后端、prompt-cache 状态下配对、交错计时。加速不来自换协议、减少动作步数、减分辨率或只计局部算子。
- 完整 predict_action 计时涵盖在线打分、索引、打包/池化、传输及模型输出；分别报告 warm、首次调用/graph capture、含 IPC 与完整 episode 时延。重型离线 profiling 不放进推理计时，但所有在线成本必须放入。
- 给出重复数、分位数/波动、资源占用、实际 Q/KV/token 数。不得删除不利样本或只引用历史 Dense 时间。

## 11. 有界闭环与 SR 判断

- 先预声明一个小型匹配 pilot 的候选与 episode 集合；先说清耗时和资源再启动，沿用 `dreamwam-release-v1` 与 CPU OSMesa，不改 success 规则。
- 成功只由 LIBERO/evaluator 判断；错误、超时、渲染问题与真正任务失败分开，覆盖不完整不报告完整 SR。
- 后续扩大规模要事先说明；不能把反复用于筛选的样本称作 held-out confirmation。小 pilot 通过不等于 SR 保持。
- 用户尚未给出“SR 没有太低”的数值阈值。请在定量验收前询问允许的相对 Dense 下降；在得到答复前报告配对结果与区间，不自行宣布 SR 达标，也不因此停止安全的离线筛选。

## 12. 交付与完成标准

- 持续维护方法、计划/进度、数据来源、实验矩阵、全部结果及负结果。每一项都有命令、commit、hash、时间、退出状态和 artifact 路径；大型原始数据不进 Git。
- 交付可运行实现、测试、冻结候选配置、完整 profiling raw data/分析报告、同输入性能结果、有限闭环证据和复现入口，并及时 add/commit/push/pull。
- 明确判断新信号是否优于现有均匀方案，是否真的 >1.5×，速度来自哪个机制，SR 证据够不够。复用已有 2× baseline 本身不算新选择规则验证成功。
- 若未达到目标，诚实给出 Pareto 前沿、失败原因及下一步；不能只凭完成代码就宣布 goal achieved。实验结束释放本任务资源，不碰其他人的进程，不遗留无人照看的无限队列。
