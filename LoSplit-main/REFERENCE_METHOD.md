# ABP-Sparse 参考判别版：研究方案与数学说明

日期：2026-11-24。对应实现：models/ABP_Reference.py、reference_training.py、reference_experiment.py。

2026-11-26 补充：SPEAR 的两份固定快照揭示了现有结构候选的覆盖限制。当前参数范围内训练 Recall 为 0%，放宽标量阈值后的结构上限为 5%。完整核对、共同参数比较及证明见 [SPEAR 诊断报告](SPEAR_ABP_DIAGNOSIS.md)。

## 资料说明

本文依据当前实现、已有开发记录、本地论文正文和本轮程序检查重新组织。实验状态以 runs/reference_ablation_20261124/study.json 和 runs/reference_backbones_20261124/study.json 为准。本文采用 CA、ASR、Precision、Recall、FPR 五项效果指标；模型数量、样本人数和运行时间作为实验配置与计算成本记录。

## 1. 研究问题与贡献定位

研究问题是：对含有异常附加分支的图，能否通过训练节点的连通关系和节点特征分布识别需要处理的接收节点，在保留正常信息的同时降低这些节点被判为指定类别的比例？

方法包含三个连续步骤。第一步根据训练节点的连通关系定位传入方向，并用特征有效维数筛选异常发送节点。第二步仅删除通过筛选的传入方向，分别训练普通分类模型和参考分类模型。第三步从留出的已标注节点计算类别分数阈值，仅在检出节点上采用参考判别。

当前可论证的贡献集中在“结构定位、特征筛选、局部参考判别”这一具体组合，以及相应的数据使用规范和结构分析。有效维数、留出校准、顺序统计量和模型集成均有成熟方法基础。文章应说明它们在本任务中的具体作用，并用消融结果识别新增步骤的贡献。

本地逐项比较已形成明确的方法差异；论文级创新强度仍需跨数据集结果、匹配模型数量的基线和组件消融共同支持。新颖性对照见 REFERENCE_AUDIT.md。

## 2. 数据与标签

设观测图为 \(G=(V,E,X)\)。原始无向边用两个消息传递方向表示，权重大于零的记录参与图处理；连通关系在对应的无向图上计算。

- \(\mathcal L\)：方法可使用的全部训练标签对应的节点身份。
- \(\mathcal V\)：原验证节点集合移除与 \(\mathcal L\) 重叠的身份后得到的集合。
- \(\widetilde y_v\)：方法可访问的记录标签，定义在 \(\mathcal L\cup\mathcal V\) 上。
- \(\mathcal C\)：干净测试集合。
- \(\mathcal T\)：另一组用于指定类别评价的测试节点。
- \(y_v\)：最终评分使用的真实标签。
- \(t\)：实验评价指定类别。
- \(\widehat t\)：方法根据检出训练节点的记录标签推断的类别。

训练、验证和测试节点的身份关系在拟合前检查。程序先合并快照中所有提供训练标签的节点，再将合并后的集合交给方法。方法接收图、记录标签和训练、验证身份；评分阶段接收真实节点名单、真实标签和评价指定类别。

“参考标签”指留出参考节点已有的记录标签。其用途是判断该节点是否属于 \(\widehat t\)，从而构造阈值所需的参考分数集合。参考集合的质量取决于原记录标签和检测后的筛选质量。

当前协议提供训练标签和验证标签，参考阈值还使用清理后验证集合的记录标签。参考分类模型采用固定训练轮数。普通分类模型使用验证准确率选择训练轮次。完整图的无标签特征与边用于阈值计算，属于预先说明的传导式图使用方式。与其他论文比较时，应同时对齐标签预算和图可见范围。

## 3. 方法

### 3.1 结构定位

对消息方向 \(u\to v\)，从图中临时移除接收节点 \(v\)，考察发送节点 \(u\) 所在连通分量是否包含训练节点。定义

\[
B(G,\mathcal L)=
\left\{u\to v:
u\ne v,\ 
\operatorname{Comp}_{G\setminus\{v\}}(u)\cap\mathcal L=\varnothing
\right\}.
\]

该集合由现有 ABP 的点双连通分解实现计算。自连接保留。结构条件表达的是：发送方与已有训练信息的连接依赖当前接收节点。

### 3.2 特征筛选

对特征向量 \(x\)，定义有效维数

\[
r(x)=
\begin{cases}
\dfrac{\|x\|_1^2}{\|x\|_2^2},&x\ne0,\\
0,&x=0.
\end{cases}
\]

使用已标注节点身份组成特征参考池
\(\mathcal P=\mathcal L\cup\mathcal V\)，计算

\[
\tau=\gamma\max_{v\in\mathcal P}r(x_v),\qquad \gamma\ge1.
\]

特征参考池仅使用节点特征。令

\[
A(G)=\{u\to v\in B(G,\mathcal L):r(x_u)>\tau\},
\quad
D(G)=\{v:\exists u,\ u\to v\in A(G)\}.
\]

数值实现使用 \(r(x_u)>\tau(1+10^{-12})\) 处理浮点比较。清理后的有向图为

\[
Q(G)=(V,E\setminus A(G),X).
\]

只移除通过结构和特征两项条件的传入方向。反向边按自身条件处理。

### 3.3 训练集合与参考集合

在训练图上得到 \(D_{\mathrm{train}}\)，定义

\[
\mathcal L_0=\mathcal L\setminus D_{\mathrm{train}},\qquad
\mathcal V_0=\mathcal V\setminus D_{\mathrm{train}}.
\]

从 \(\mathcal L_0\) 中按固定随机种子均匀留出

\[
m=\lceil\rho|\mathcal L_0|\rceil,\qquad 0<\rho<1
\]

个节点，得到 \(\mathcal R_{\mathrm{train}}\)。固定人数选项直接指定 \(m\)，用于复现已有的 120 节点方案。参考分类模型的训练集合和阈值参考集合分别为

\[
\mathcal F=\mathcal L_0\setminus\mathcal R_{\mathrm{train}},
\qquad
\mathcal R=\mathcal R_{\mathrm{train}}\cup\mathcal V_0.
\]

普通模型在 \(Q(G)\) 上使用 \(\mathcal L_0\) 的标签训练，并依据 \(\mathcal V_0\) 的准确率选择轮次。参考模型使用 \(\mathcal F\) 的标签，固定训练 \(E_{\mathrm{train}}\) 轮；训练时额外移除与 \(\mathcal R_{\mathrm{train}}\) 关联的边，阈值计算和预测时恢复这些节点在 \(Q(G)\) 中的连接。

模型结构可以选择图卷积网络（GCN）、图注意力网络（GAT）或 GraphSAGE。各结构沿用项目现有的逐层计算和训练设置。

两类模型的训练损失均为对应训练集合上的平均负对数似然：

\[
\mathcal J(\theta;\mathcal S)
=-\frac1{|\mathcal S|}\sum_{v\in\mathcal S}
\log p_\theta(\widetilde y_v\mid Q(G),v).
\]

参考模型的图输入还包含上述参考训练节点的删边操作。每类模型分别训练 \(M\) 个成员，成员随机种子依次为 \(s,s+10,\ldots,s+10(M-1)\)。默认 \(M=3\)，共训练六个模型。结构与特征检测器采用确定性计算。

### 3.4 推断类别

在检出的训练节点中统计记录标签：

\[
\widehat t=
\arg\max_c
\sum_{v\in D_{\mathrm{train}}\cap\mathcal L}
\mathbf1\{\widetilde y_v=c\}.
\]

计数相同则选择类别编号较小者。检出训练节点集合为空时，最终使用普通模型的平均概率预测，并记录这一分支。

### 3.5 参考阈值

对第 \(j\) 个参考模型，定义类别分数

\[
s_j(v)=\log p_j(\widehat t\mid v)
-\max_{c\ne\widehat t}\log p_j(c\mid v).
\]

阈值参考集合中取

\[
\mathcal R^-=\{v\in\mathcal R:\widetilde y_v\ne\widehat t\},
\qquad n=|\mathcal R^-|.
\]

将 \(n\) 个分数升序排列为
\(s_{j,(1)}\le\cdots\le s_{j,(n)}\)。给定排序参数 \(\alpha\)，设

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad
\delta_j=
\begin{cases}
\max\{0,s_{j,(k)}\},&k\le n,\\
+\infty,&k=n+1.
\end{cases}
\]

\(n=0\) 时程序要求补充有效参考节点。样本数导致 \(k=n+1\) 时使用扩展顺序统计量 \(+\infty\)，并在元数据中记录。最大值规则可以直接令 \(k=n\)；已有辅助函数保留这一选项。

### 3.6 最终预测

普通模型和参考模型的平均概率分别为

\[
\overline p^{\,o}_v(c)=\frac1M\sum_{j=1}^M p^o_j(c\mid v),
\qquad
\overline p^{\,r}_v(c)=\frac1M\sum_{j=1}^M p^r_j(c\mid v).
\]

在当前输入图重新计算 \(Q(G)\) 和 \(D(G)\)。最终输出

\[
\widehat y_v=
\begin{cases}
\arg\max_c\overline p^{\,o}_v(c),&v\notin D(G),\\
\widehat t,&v\in D(G),\ \forall j:s_j(v)>\delta_j,\\
\arg\max_{c\ne\widehat t}\overline p^{\,r}_v(c),&\text{其余情况}.
\end{cases}
\]

类别判断采用严格大于，等于阈值时执行第三行。每个节点均返回一个类别。

## 4. 参数数量与跨数据集设置

### 4.1 四个方法参数

| 参数 | 作用 | 当前启动默认值 | 消融水平 |
|---|---|---:|---|
| \(\gamma\) | 特征阈值倍数 | 2 | 1、2、3 |
| \(\rho\)，或固定 \(m\) | 从保留训练集合留出的比例或人数 | 0.2 | 0.1、0.2、0.3，另复现 \(m=120\) |
| \(\alpha\) | 参考分数的排序位置 | 0.005 | 0.001、0.005、0.009、0.02 |
| \(M\) | 每类模型的成员数 | 3 | 1、3、5 |

\(\rho\) 与 \(m\) 控制同一个取样规模；固定人数参数覆盖比例参数。数值阈值 \(\tau,\delta_j\) 和实际人数 \(m,n\)由数据计算。

另有五项共享训练设置：训练轮数、隐藏维数、学习率、dropout、权重衰减。默认分别为 200、32、0.01（DPGBA 配对快照使用 0.015）、0.5、0.0005。模型结构、两个随机种子和组件开关记录在实验配置中。参考版的附加训练惩罚权重固定为零。

### 4.2 120 的来源

历史 reference_protocol.json 明确记录了在开发图上比较 120、160、200 后固定 120。risk_buffer_protocol.json 则记录了依据开发阶段 ASR 反馈将 \(\alpha\) 改为 0.005 的过程。因此 120 和最终排序规则都具有经验选择来源。

新增比例模式按 \(|\mathcal L_0|\) 计算人数。分母使用检测后保留的训练节点，因为这些节点同时承担监督训练和参考抽样预算。全图节点数会随未标注节点规模改变，用它设置参考人数容易使参考需求超过可用训练标签。

0.2 是本轮明确登记的起始配置，跨数据集实验按同一比例进行比较。固定 120 选项保留，历史结果继续按原配置说明。以当前一份 Cora 快照为例，保留训练节点数为 541 时，0.2 对应 109 个留出节点。

### 4.3 排序分辨率

有限阈值要求

\[
n\ge\left\lceil\frac1\alpha\right\rceil-1.
\]

例如 \(\alpha=0.005\) 对应至少 199 个非推断类别参考节点；\(\alpha=0.001\) 对应至少 999 个。这里的人数包含训练留出节点与清理后的验证节点，并经过类别筛选。

在 \(199\le n\le398\) 时，\(\alpha=0.005\) 对应样本最大值。多个相邻 \(\alpha\) 可能产生相同整数排序位置，因此实验同时记录 \(n\)、阈值与无限阈值标记。

更换 Citeseer、Pubmed 时，先保持同一比例和其他方法参数，报告实际 \(m,n\) 以及计算成本，再使用独立开发实验评估调整需要。最终比较使用预先固定的配置。

## 5. 数学性质

### 命题一：有效维数的尺度性质

若 \(x\ne0\)，记非零分量数为 \(q\)，则

\[
1\le r(x)\le q\le d,\qquad r(ax)=r(x)\quad(a\ne0).
\]

证明：令 \(z_i=|x_i|\)。展开平方得到
\((\sum_i z_i)^2=\sum_i z_i^2+2\sum_{i<j}z_iz_j\ge\sum_i z_i^2\)，故下界成立。由柯西—施瓦茨不等式，
\((\sum_{i:z_i>0}z_i)^2\le q\sum_i z_i^2\)，得到上界。将 \(ax\) 代入定义后，分子分母均乘 \(a^2\)，约去即得尺度性质。证毕。

### 命题二：附加分支满足条件时，原节点的固定模型输出保持一致

考虑原图 \(G_0\) 及增加分支后的图 \(G_1\)。假设：

1. 新增分支通过单个原节点接入；新增分支内部含有零个训练节点。
2. 两图使用相同的训练身份、特征参考池、原节点特征及原边权。
3. 所有新增到原图的传入方向均属于 \(A(G_1)\)。
4. 原图方向在两图中的保留状态相同。
5. 两次计算使用相同模型参数和评估模式；聚合依赖保留的传入邻居及固定自连接。

则对当前 GCN、GAT 和 GraphSAGE 的任意原节点 \(v\)，清理图上的逐层表示一致。

证明：第零层表示由原节点特征给出，因此一致。清理后，原节点收到的消息均来自相同原节点集合，原边权相同。对于 GCN，实现中的度数由传入边计算，所有原节点的相关度数和归一化系数一致。对于 GAT，接收端的邻居集合与自连接一致，上一层表示一致，因此注意力分数、归一化分母及加权和一致。对于 GraphSAGE，邻居平均值与自身变换分别一致。若第 \(\ell\) 层原节点表示一致，则上述聚合和固定的逐节点变换给出第 \(\ell+1\) 层一致。由归纳法得到所有层及分类输出一致。证毕。

单接入且无训练节点的附加分支符合结构定位的条件；特征筛选是否覆盖其全部传入方向由第三项假设明确规定。程序测试验证了三种算子的该性质。该命题比较固定参数下的图输入，最终局部判别仍按检出集合和阈值执行。

### 命题三：参考标签与参考模型参数的分离

固定公开类别集合、图、节点划分、模型结构、初始参数、随机状态、优化器和训练轮数。参考分类模型只在 \(\mathcal F\) 上计算损失，并按固定轮数保存。改变 \(\mathcal R\) 上的记录标签时，参考模型训练后的参数保持一致。

证明：第零轮参数与优化器状态固定。第 \(e\) 轮的模型输出由当前参数、图、特征和该轮随机状态确定；损失只读取 \(\mathcal F\) 上的标签。因此改变 \(\mathcal R\) 标签保持该轮损失、梯度、参数更新和优化器状态一致。由轮次归纳得到最终参数一致。固定轮数的保存规则保持这一结论。证毕。

参考标签随后参与阈值计算，所以阈值及最终参考判别会随这些标签改变。普通模型按其独立训练规则使用训练和验证标签。

### 命题四：交换性条件下的单模型排序界

给定拟合模型、推断类别与必要的条件信息，假设 \(n\) 个参考分数和一个未来节点分数
\(S_1,\ldots,S_n,S_{n+1}\) 来自同一非推断类别总体，且具有交换性。参考记录标签在此命题中正确标识该总体。则上述阈值满足

\[
\Pr(S_{n+1}>\delta_j)
\le\frac{n+1-k}{n+1}\le\alpha
\quad(k\le n).
\]

当 \(k=n+1\) 时接受概率为零。

证明：先用独立随机次序打破并列分数，使 \(n+1\) 个分数的排序位置具有对称性。交换性使未来分数的排序位置均匀分布在 \(1,\ldots,n+1\) 上。事件“未来分数严格大于参考第 \(k\) 小分数”包含于“未来分数排序位置大于 \(k\)”中，因此其概率至多为 \((n+1-k)/(n+1)\)。将阈值提升到与零的最大值只会减少接受事件。由 \(k\ge(n+1)(1-\alpha)\)，得到第二个不等式。无限阈值的接受事件为空。证毕。

图上的训练、结构筛选和参考抽样都会影响分数分布。若希望把本命题用于检出测试节点，应在“测试节点被检出”这一条件下验证相应交换性。当前实验使用实际 CA、ASR、Precision、Recall、FPR 评价结果；本命题提供条件明确的排序解释。

### 命题五：多成员共同接受

设 \(A_j=\{S_j>\delta_j\}\)。联合规则的接受事件为
\(A=\bigcap_{j=1}^M A_j\)。对每个 \(j\) 有 \(A\subseteq A_j\)，故

\[
\Pr(A)\le\min_j\Pr(A_j).
\]

证明：对事件包含关系直接应用概率的单调性。证毕。该结果允许成员之间相关；若每个成员都满足上一命题的同一条件，则共同接受概率也至多为 \(\alpha\)。

### 命题六：局部替换的 CA 变化与 ASR 分解

固定同一清理图上的普通模型预测 \(f\) 和最终预测 \(h\)。令 \(N_{\mathrm{gain}}\) 为干净测试集合中由错误变正确的人数，\(N_{\mathrm{loss}}\) 为由正确变错误的人数。两种变化均发生在 \(D(G)\cap\mathcal C\) 内，因此

\[
\operatorname{CA}(h)-\operatorname{CA}(f)
=\frac{N_{\mathrm{gain}}-N_{\mathrm{loss}}}{|\mathcal C|},
\quad
\left|\operatorname{CA}(h)-\operatorname{CA}(f)\right|
\le\frac{|D(G)\cap\mathcal C|}{|\mathcal C|}.
\]

证明：在未被标记的节点上，两种预测一致；在其余节点上，正确性指示量之差取 \(1,-1,0\)。逐节点求和即可。证毕。

令 \(\mathcal S=\{v\in\mathcal T:y_v\ne t\}\)，将其分为检出部分和漏检部分，人数分别为 \(n_D,n_M\)。对应指定类别预测人数为 \(a_D,a_M\)，则

\[
\operatorname{ASR}
=\frac{a_D+a_M}{n_D+n_M}.
\]

两部分人数均为正时，可写为

\[
\operatorname{ASR}
=\frac{n_M}{|\mathcal S|}\frac{a_M}{n_M}
+\frac{n_D}{|\mathcal S|}\frac{a_D}{n_D}.
\]

证明：两个集合互不相交且并集为 \(\mathcal S\)，指定类别预测的计数可直接相加。证毕。空集合按零计数参与第一式，单独条件比例记为缺失值。该式把检测覆盖和检出后的类别判断联系起来。

## 6. 五项评价指标

### 干净准确率

\[
\operatorname{CA}
=\frac{\sum_{v\in\mathcal C}\mathbf1\{\widehat y_v=y_v\}}{|\mathcal C|}.
\]

### 指定类别预测率

\[
\operatorname{ASR}
=\frac{\sum_{v\in\mathcal S}\mathbf1\{\widehat y_v=t\}}{|\mathcal S|},
\qquad
\mathcal S=\{v\in\mathcal T:y_v\ne t\}.
\]

### 检测精确率、召回率与误报率

主检测总体统一取 \(\mathcal U=\mathcal L\)。令 \(\mathcal B\subseteq\mathcal U\) 为评分阶段使用的真实目标节点集合，\(\widehat{\mathcal B}=D_{\mathrm{train}}\cap\mathcal U\)。定义

\[
\mathrm{TP}=|\widehat{\mathcal B}\cap\mathcal B|,\quad
\mathrm{FP}=|\widehat{\mathcal B}\setminus\mathcal B|,\quad
\mathrm{FN}=|\mathcal B\setminus\widehat{\mathcal B}|,\quad
\mathrm{TN}=|\mathcal U\setminus(\widehat{\mathcal B}\cup\mathcal B)|.
\]

\[
\operatorname{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad
\operatorname{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad
\operatorname{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}.
\]

JSON 保存整数计数与原始比例，表格显示百分数。分母为零时输出 null，表格显示“—”。CA 与 ASR 的分母为空时实验停止并给出原因。参考分类的错误接受概率和检测 FPR 分别按各自定义记录。

## 7. 实验设计

本轮在同一批保存快照上进行配对实验。四个方法参数分别改变一个，其余配置固定。另加入固定 120 人、参考抽样种子变化，以及下列三种组件消融：

1. 只使用清理图上的普通模型，移除参考判别。
2. 用完整 ABP 投影替换选择性删边。
3. 保留检出训练节点的记录标签参与模型训练。

各模型使用同样的训练轮数、隐藏维数和种子安排。GCN、GAT、GraphSAGE 分别训练、保存并重载检查。成员数消融同时报告实际模型数量；正式基线应匹配普通模型的成员数，另列完整方法的额外参考模型成本。

跨数据集确认使用 Cora、Citeseer、Pubmed，分别记录实际参考人数与非推断类别人数。每个数据集建议使用至少三个重新生成的图设置和三个模型种子组。当前已参与开发比较的图记录为开发数据；最终确认采用预先保留的新设置。

消融记录全部配置，最终参数选择依据预先规定的开发流程。固定图划分下的不同种子重复表示该划分内的训练与图生成变异；跨划分实验单独报告。当前主结果表保留五项效果指标。

## 8. 使用方式

在 run_experiments.py 顶部 SETTINGS 中设置：

~~~python
"stage": "defense",
"methods": ["abp_reference"],
"dataset": "Cora",
"model": "GCN",
"abp_reference_multiplier": 2.0,
"abp_reference_ratio": 0.2,
"abp_reference_count": None,
"abp_reference_risk": 0.005,
"abp_reference_members": 3,
"abp_reference_seed": 20261124,
~~~

attack_root 指向同次运行保存的快照和配对模型所在目录；defense_root 使用新目录。固定人数复现设置 abp_reference_count=120。切换模型使用 GAT 或 GraphSage。通用入口先执行 dry-run 检查路径和参数：

~~~text
python run_experiments.py --stage defense --methods abp_reference --model GAT --dry-run
python run_experiments.py --stage defense --methods abp_reference --model GraphSage
python run_experiments.py --stage defense --methods abp_reference --abp-reference-count 120
~~~

完整消融入口为 run_reference_ablation.py。先建立实验清单，再增加 execute 参数运行：

~~~text
python run_reference_ablation.py --input-root "runs/comparison_inputs_GCN4/{dataset}/{model}/seed{seed}" --output-root runs/reference_plan --datasets Cora Citeseer Pubmed --models GCN GAT GraphSage --seeds 17 18 19 --classifier-seeds 10 40 70 --cases base
python run_reference_ablation.py --input-root "runs/abp_sparse_20261123/seed{seed}/inputs" --output-root runs/reference_ablation_new --datasets Cora --models GCN --seeds 16 --workers 2 --execute
~~~

第一次命令只生成清单；执行时使用新的输出目录。已有结果保留，失败日志与配置一并保存。模型包包含结构名称、特征维数、类别数、两组权重、检测器、参考阈值和训练身份，可通过 load_reference_bundle 重新加载。

本机 PyTorch 2.0.0 的 GAT 分组最大值算子在严格确定性 CUDA 模式下缺少实现。参考入口实际探测该算子，遇到这一情况时记录设备变化并使用 CPU。GCN 和 GraphSAGE 已通过当前 GPU 的严格模式入口。显式使用 CPU 可添加 reference-device 参数；GAT 使用常规 GPU 运算可关闭严格确定性：

~~~text
python run_experiments.py --stage defense --methods abp_reference --model GAT --reference-device cpu
python run_experiments.py --stage defense --methods abp_reference --model GAT --no-deterministic
~~~
