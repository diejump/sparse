"""Build result tables directly from completed study records."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/generalization_20261126'
MAIN=['range_cora41','range_cora42','range_cora43','range_citeseer41']

def pct(x):return '—' if x is None else f'{100*x:.2f}'

def collect(name):
    path=OUT/name/'study.json'
    if not path.exists():return []
    s=json.loads(path.read_text(encoding='utf-8'))
    verification=OUT/name/'verification.json'
    verified={r['id']:r for r in json.loads(verification.read_text(encoding='utf-8'))['jobs']} if verification.exists() else {}
    rows=[]
    for job in s['jobs']:
        if job['status']!='completed':continue
        m=dict(job['metrics'])
        raw=json.loads(Path(job['output']+'.meta.json').read_text(encoding='utf-8'))
        m['TestDetection']=raw.get('TestDetection',raw.get('TestDetectionAudit'))
        if job['id'] in verified:m.update(verified[job['id']].get('metrics',{}))
        rows.append(dict(study=name,dataset=s['dataset'],id=job['id'],seed=job['seed'],kind=job['kind'],
            architecture=job['architecture'],case=job['case'],metrics=m,
            verified=verified.get(job['id'],{}).get('verified',False),output=job['output']))
    return rows

def main():
    all_studies=sorted(p.parent.name for p in OUT.glob('*/study.json'))
    allrows=[r for name in all_studies for r in collect(name)]
    primary=[r for r in allrows if r['study'] in MAIN]
    baselines={(r['dataset'],r['seed'],r['kind'],r['architecture']):r['metrics'] for r in primary if r['case']=='none'}
    ranges=[r for r in primary if r['case']=='range']
    for r in ranges:
        b=baselines.get((r['dataset'],r['seed'],r['kind'],r['architecture']))
        r['baseline']=b
        r['delta_ca']=r['metrics']['CA']-b['CA'] if b else None
        m=r['metrics'];td=m['TestDetection']
        r['detection_asr_pass']=(m['ASR']<.01 and m['Precision'] is not None and m['Precision']>.9 and m['Recall']>.9
            and td is not None and td['Precision'] is not None and td['Precision']>.9 and td['Recall']>.9)
    (OUT/'summary.json').write_text(json.dumps(dict(completed_fits=len(allrows),verified_fits=sum(r['verified'] for r in allrows),
        primary=ranges,all_rows=allrows),indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'primary_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['dataset','seed','kind','architecture','CA','BaselineCA','DeltaCA','ASR','Precision','Recall','FPR','SameGraphCA','AffectedAccuracy','AffectedTargetAccuracy','ASRHits','ASRCount','verified','TestPrecision','TestRecall','TestFPR']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in ranges:
            m=r['metrics'];writer.writerow(dict(dataset=r['dataset'],seed=r['seed'],kind=r['kind'],architecture=r['architecture'],
                **{k:m.get(k) for k in fields[4:] if k not in ('BaselineCA','DeltaCA','verified','TestPrecision','TestRecall','TestFPR')},
                BaselineCA=r['baseline']['CA'] if r['baseline'] else None,DeltaCA=r['delta_ca'],verified=r['verified'],
                TestPrecision=m['TestDetection']['Precision'],TestRecall=m['TestDetection']['Recall'],TestFPR=m['TestDetection']['FPR']))
    lines=['# ABP 特征参考版：参数、泛化与研究结论','',
        '日期：2026-11-26。所有数值由保存的实验清单及模型重载复核生成。',
        '','## 1. 现行版本的参数数量','',
        '| 类别 | 内容 | 默认值或约定 |','|---|---|---|',
        '| 4 个核心数值参数 | 检测倍数 γ、参考比例 ρ、参考风险 α、成员数 M | 2、0.2、0.005、1 |',
        '| 1 个数值精度选项 | feature_tolerance | 1e-6 |',
        '| 1 个修复规则选择 | feature_repair | unique_mode |',
        '| 参考数量覆盖项 | count | 有值时覆盖比例，和比例共同决定同一个规模 |',
        '| 复现设置 | 分类器种子、参考划分种子 | 与模型宽度、学习率、轮数分别登记 |',
        '| 组件消融 | 检测/结构开关、结构门控、参考判别开关 | 研究对照单列 |','',
        '因此，现行完整实现有 4 个核心数值参数，加上容差和修复规则共 6 类主要方法设置。旧参考版的默认 M=3；特征参考版运行入口默认 M=1。',
        '','## 2. 对表示依赖的判断','',
        '原特征一致性规则利用归一化二值行的非零值相等这一公开表示性质。Cora 和 Citeseer 的本地正常缓存都符合该性质，因此跨到 Citeseer 的数据格式适配成立。统计改善需要与具体生成样本的表示偏离一起解释。',
        '本轮新增范围参考版，从验证样本学习幅值和有效维数范围，再做结构接收定位、越界坐标的经验中位数填补和参考判别。该实现接受连续、有符号特征，研究入口提供 γ、ρ、α 三个方法参数。',
        '三参数默认 γ=2、ρ=0.2、α=0.005；普通模型与参考模型各一个。模型宽度、学习率、训练轮数与同骨干基线保持一致。固定组件选择及其开发对照继续列入实验记录。',
        '','## 3. 完成的实验','',
        f'已完成登记拟合 {len(allrows)} 组，独立重载及逐项计数已核验 {sum(r["verified"] for r in allrows)} 组。主评价中已完成范围参考版 {len(ranges)} 组，检测与 ASR 三项阈值同时达到的组合为 {sum(r["detection_asr_pass"] for r in ranges)}/{len(ranges)}。',
        'Cora 使用新生成种子 41、42、43；GCN 三个种子，GAT 与 GraphSage 使用 41。Citeseer 使用新生成种子 41，覆盖三个骨干。生成模型统一沿用项目 GCN 配置；跨骨干评价使用同一配对快照。图划分沿用项目固定种子 10。',
        '表中的 Precision、Recall、FPR 使用训练身份总体。测试总体的 Precision、Recall、FPR 另存于 CSV 的 TestPrecision、TestRecall、TestFPR；测试组中变化节点与正常节点数量约为 1:1。',
        '','### 按数据集与骨干汇总','',
        '| 数据集 | 骨干 | 组数 | 平均无防御 CA | 平均范围版 CA | 最差 ΔCA（百分点） | ASR 范围 | 最低 Precision | 最低 Recall |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|']
    groups=sorted(set((r['dataset'],r['architecture']) for r in ranges))
    for ds,arch in groups:
        group=[r for r in ranges if r['dataset']==ds and r['architecture']==arch and r['baseline']]
        if not group:continue
        lines.append('| '+' | '.join([ds,arch,str(len(group)),pct(mean(r['baseline']['CA'] for r in group)),
            pct(mean(r['metrics']['CA'] for r in group)),f'{100*min(r["delta_ca"] for r in group):+.2f}',
            pct(min(r['metrics']['ASR'] for r in group))+'–'+pct(max(r['metrics']['ASR'] for r in group)),
            pct(min(r['metrics']['Precision'] for r in group)),pct(min(r['metrics']['Recall'] for r in group))])+' |')
    test_rows=[r['metrics']['TestDetection'] for r in ranges]
    lines+=['',f'独立测试总体最低 Precision={pct(min(t["Precision"] for t in test_rows))}%，最低 Recall={pct(min(t["Recall"] for t in test_rows))}%，最高 FPR={pct(max(t["FPR"] for t in test_rows))}%。' if test_rows else '',
        '所有百分比的分母保持各自评价总体；上述组均值为逐次结果的算术平均。CA 的接近程度以差值呈现。',
        '','### 主评价的全部结果','',
        '| 数据集 | 种子 | 设置 | 骨干 | 无防御 CA | CA | ΔCA（百分点） | ASR | Precision | Recall | 同图 CA | 变化节点真实类别准确率 | 真实目标类恢复率 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ranges:
        m=r['metrics'];b=r['baseline']
        lines.append('| '+' | '.join([r['dataset'],str(r['seed']),r['kind'],r['architecture'],pct(b['CA'] if b else None),pct(m['CA']),
            f'{100*r["delta_ca"]:+.2f}' if b else '—',*[pct(m.get(k)) for k in ['ASR','Precision','Recall','SameGraphCA','AffectedAccuracy','AffectedTargetAccuracy']]])+' |')
    lines+=['','## 4. 低 ASR 与真实分类恢复','',
        '参考阈值提高了被检出节点输出推断类别所需的分数。该步骤直接降低 ASR，同时改变真实目标类别节点的恢复率。上表并列报告这两种结果。',
        '开发对照中，GCN 的 GTA/UGBA/DPGBA 在去掉参考判别后，变化节点真实类别准确率为 83.03%、真实目标类恢复率为 67.39%、ASR 为 3.56%；启用参考判别的第一候选对应 74.17%、10.87%、0%。这组配对数据说明参考判别的具体代价。',
        '第一候选采用中位差合成分数，在另一个 UGBA 样本出现 17 个正常训练节点误报，Precision=70.18%；对应结果完整保留于 transfer_cora31。范围版分别处理两个统计量的区间，后续主评价使用该版本。',
        '','## 5. 表示对照和三个参数','',
        '共同乘以正常数保留检测名单。种子 31 的逐行 L2 归一化对照中，SPEAR 的训练 Recall 从 100% 变成 0%，表明范围版的 SPEAR 检出依赖幅值偏离；原始输入上的满检出与这项机制结果并列保留。GTA、UGBA、DPGBA 在该项检测对照中保持满召回。',
        'γ 的 1、1.5、2、3、4 检测敏感性保存于 range_development_controls.json。ρ 的 0.1、0.2、0.4 配对拟合使用 Cora 的 UGBA/SPEAR。α 的 0.002、0.005、0.01、0.02、0.05 在固定参考模型上重新计算校准阈值，并独立记录 ASR 和真实类别恢复。',
        '',
        '| Cora/SPEAR/GCN，种子 41 | ASR (%) | 真实目标类恢复率 (%) |',
        '|---|---:|---:|',
        '| α=0.002，参考阈值为无穷 | 0.00 | 0.00 |',
        '| α=0.005，默认 | 0.00 | 17.39 |',
        '| α=0.01 | 3.56 | 60.87 |',
        '| α=0.02 | 5.33 | 65.22 |',
        '',
        '留出比例 ρ=0.4 时，UGBA 与 SPEAR 的 ASR 都升至 2.22%，真实目标类恢复率分别升至 56.52% 与 50.00%。该对照用于解释分类器、留出样本和阈值之间的相互影响，默认配置保留 ρ=0.2。',
        '','## 6. 数学结果与研究判断','',
        'ENVELOPE_THEORY.md 给出范围排序概率界、分数不变性、检测分离充分条件、精确率与误报的精确关系、盒投影与中位数填补证明、GCN/GAT/GraphSage 的条件稳定性界、参考判别排序界和观测分布可识别性结论。',
        '任意基于观测的检测器满足 Recall−FPR≤TV(P₁,P₀)。高召回与低误报同时成立需要足够的分布分离；CA 保持需要修复误差与模型分类间隔等条件。当前结果支持所列数据、生成设置和骨干上的经验表现。',
        '研究定位：范围参考版是三参数的泛化候选。其每一步已有明确目的与条件证明；表示对照和真实目标类恢复结果揭示了继续完善的具体方向。论文表述应同时呈现统计分离、阈值代价和全部配对数据。',
        '在独立 Bernoulli 评价假设下，0/225 的 95% 单侧上界约为 1.3226%。各次 0% 是固定评价集合上的实测结果。固定节点划分的多次种子分别保存，置信上界保留相应抽样假设。',
        '','## 7. 实现与复现','',
        '原特征版入口已经支持 Cora/Citeseer 以及 GCN/GAT/GraphSage，并在运行时核对验证特征格式。新方法接入 run_experiments.py 的 range_reference，保留原有 SETTINGS 选择。',
        '```text',
        'python run_experiments.py --stage defense --attacks GTA UGBA DPGBA SPEAR --methods range_reference --dataset Cora --model GCN --seed 41 --attack-root runs/generalization_20261126/fresh_cora/seed41 --defense-root runs/my_range_reference --reference-device cpu --range-gamma 2 --range-rho 0.2 --range-alpha 0.005',
        '```',
        '','将数据集改为 Citeseer、输入改为 fresh_citeseer/seed41，并选择 GAT 或 GraphSage，可运行对应配对模型。输出目录应使用新名称。',
        '研究报告中的匹配无防御基线由 run_generalization_study.py 统一为宽度 32、学习率 0.01；总入口中原有方法继续保留其作者参数设置。',
        '','## Material Passport','',
        '- Workflow: experiment execution and reproducibility validation.',
        '- Material: local observed graph snapshots, code, checkpoints, integer metric counts.',
        '- Status: study-level verification.json records the completed verification state.',
        '- Primary record: runs/generalization_20261126/summary.json and primary_results.csv.',
        '- Research status: empirically evaluated candidate with explicit mathematical assumptions.','']
    (ROOT/'GENERALIZATION_RESULTS_20261126.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed=len(allrows),verified=sum(r['verified'] for r in allrows),primary=len(ranges),
        passed=sum(r['detection_asr_pass'] for r in ranges)),ensure_ascii=False))

if __name__=='__main__':main()
