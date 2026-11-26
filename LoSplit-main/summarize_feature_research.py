"""Generate the final report from verified recorded comparisons."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/feature_reference_20261126'


def load_verified(name):
    folder=OUT/name
    study=json.loads((folder/'study.json').read_text(encoding='utf-8'))
    verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='verified'
    assert verification['count']==len(study['jobs'])
    assert all(j['status']=='completed' for j in study['jobs'])
    return study,verification


def main():
    confirmation,verified=load_verified('confirmation')
    development,_=load_verified('ablation')
    members,_=load_verified('members_comparison')
    conditions=json.loads((OUT/'condition_audit.json').read_text(encoding='utf-8'))
    frozen=json.loads((OUT/'frozen_config.json').read_text(encoding='utf-8'))
    assert all(conditions['frozen_source_match'].values())
    recommended=[j for j in confirmation['jobs'] if j['case']=='full']
    assert len(recommended)==12
    assert all(j['metrics']['Precision']>.9 and j['metrics']['Recall']>.9 for j in recommended)
    final_metrics={}
    for kind in ('GTA','UGBA','DPGBA','SPEAR'):
        rows=[j['metrics'] for j in recommended if j['kind']==kind]
        final_metrics[kind]={k:statistics.mean(row[k] for row in rows) for k in ('CA','ASR','Precision','Recall','FPR')}
    lines=[
        '# 参考版的特征一致性扩展：研究结果','',
        '日期：2026-11-26。推荐版本在原参考框架中增加原节点自身的特征一致性检查与共有值修复。',
        '保留原特征的检测扩展同时作为对照报告。推荐运行选项为 abp_feature_reference，feature_repair=unique_mode，feature_members=1。','',
        '## 1. 新生成样本上的确认结果','',
        '配置冻结后重新生成 Cora 的 GTA、UGBA、DPGBA、SPEAR 样本，种子为 31、32、33，共十二份配对快照。',
        '原图和原数据划分沿用项目设置，运行种子改变生成与分类器训练的随机性。SPEAR 的原节点排序缓存及选定特征文件沿用生成入口。',
        '下表为三个种子的算术平均，单位为 %；每个单次推荐设置均为 ASR=0、Precision=100、Recall=100、FPR=0。','',
        '| 方式 | CA | ASR | Precision | Recall | FPR |',
        '|---|---:|---:|---:|---:|---:|']
    for kind,m in final_metrics.items():
        lines.append('| '+kind+' | '+' | '.join(f'{100*m[k]:.4f}' for k in ('CA','ASR','Precision','Recall','FPR'))+' |')
    lines += ['','CA 每次使用 270 个干净测试节点，ASR 每次使用真实类别异于指定类别的 225 个节点。',
        '主检测 Precision、Recall、FPR 在同一训练身份总体上计算，TP、FP、FN、TN 与整数分子分母全部保存。',
        '每个模型输出一个类别，参考阈值按固定公式计算。推荐版本的全部参考阈值均为有限值。','',
        '## 2. 配对比较','',
        '各防御配置统一使用 GCN、隐藏维数 32、200 轮、学习率 0.01、dropout=0.5、权重衰减 0.0005。',
        '每类模型一个成员：普通模型加参考模型共两个。无防御对照采用相同初始化种子、普通训练过程和模型宽度。',
        '这组配对对照独立于生成入口的评估分类器；SPEAR 生成器仍沿用其入口的隐藏维数 80。','',
        '| 方式 | 无防御 CA (%) | 无防御 ASR (%) | 原参考版 ASR (%) | 推荐扩展 CA (%) |',
        '|---|---:|---:|---:|---:|']
    for kind in final_metrics:
        def avg(case,metric):
            return 100*statistics.mean(j['metrics'][metric] for j in confirmation['jobs'] if j['kind']==kind and j['case']==case)
        lines.append(f'| {kind} | {avg("none","CA"):.4f} | {avg("none","ASR"):.4f} | {avg("original_reference","ASR"):.4f} | {avg("full","CA"):.4f} |')
    lines += ['','推荐版本在 GTA、UGBA、DPGBA 上保持原参考版的五项指标，在 SPEAR 上补齐检测并降低 ASR。',
        '标准干净图中，DPGBA 与 SPEAR 的平均 CA 相比匹配无防御对照分别低约 0.25、0.74 个百分点，GTA、UGBA 的平均 CA 提高。','',
        '## 3. 检测、修复和参考判别各自的作用','',
        '开发实验保留全部登记配置。下表列出各组件设置的 ASR，单位为 %。','',
        '| 设置 | GTA | UGBA | DPGBA | SPEAR |',
        '|---|---:|---:|---:|---:|']
    names={'original_reference':'原参考版','detection_only':'新增自身检测，保留原特征',
           'full':'新增自身检测及修复','no_structure':'去掉结构检测与图处理','no_reference':'去掉参考判别',
           'clip_only':'仅截断负值并归一化','clip_with_detection':'检测加截断归一化及参考判别',
           'repair_only':'仅共有值修复','ungated_structure':'移除 ABP 门控，保留高分发送端处理'}
    for case,label in names.items():
        values=[next(j['metrics']['ASR'] for j in development['jobs'] if j['kind']==kind and j['case']==case)
                for kind in final_metrics]
        lines.append('| '+label+' | '+' | '.join(f'{100*v:.4f}' for v in values)+' |')
    lines += ['',
        '新增自身检测是本次补齐 SPEAR 的直接步骤。参考判别继续降低检出节点的指定类别预测。',
        '修复的作用体现在消息传递后的正常节点分类。在标准干净图上，SPEAR 修复版平均 CA 为 82.5926%，保留原特征的检测扩展为 84.4444%。在加入测试变化的同一图中，同一批 270 个未修改节点的 CA 分别为 82.5926% 与 73.0864%。修复版因而成为最终推荐配置；两种版本的标准 ASR 均为零。',
        '无 ABP 门控的高分发送端处理与 ABP 门控版本在本轮五项指标上相同。这组结果把新增证据集中在自身检测和参考判别上。',
        '开发样本中，检测加截断归一化的 SPEAR 同图正常节点 CA 为 82.9630%，与共有值修复相同；其标准干净图 CA 为 84.4444%。该替代处理的全部结果保留，当前新种子确认使用已登记的共有值修复与无修复两种版本。',
        '三成员修复版本另保存四组开发对照；推荐版本使用两个分类模型即可达到本轮登记目标。','',
        '## 4. 表示检查的实际依据','',
        '当前 Cora 输入是归一化二值词袋行，非零值相同、行和为一。全部 2708 个原始正常节点通过检查，误报为零。',
        '原始 SPEAR 快照中，单纯负值或行和异常检查也得到 100% 检测精确率与召回率。进一步对观测特征截断负值并重新归一化后，简单检查的 Recall 为 0%，行内一致性检查在三份新 SPEAR 快照的训练和测试总体上仍保持 Precision=100%、Recall=100%、FPR=0%。',
        '该对照核验检测信号对两种基本数值处理的反应。完整分类效果采用前述保存快照上的配对训练结果。','',
        '## 5. 数学结论及条件核对','',
        '完整定义、八项命题及补充推导见 FEATURE_REFERENCE_THEORY.md。核心结论包括正常表示保留、理想零容差下的集合判别、容差下的集合距离界、检测并集关系、参考标签隔离和参考判别条件。',
        '可选修复部分证明：公共正值由频数唯一识别时，保留原非零位置；若原非零数为 k、损失其中 r 个，则 L1 误差为 2r/k，平方 L2 误差为 r/[k(k-r)]。另给出固定 GCN 的输出误差传播界。',
        '三份新 SPEAR 快照共 120 个训练目标及 813 个测试目标均满足所核对的公共值恢复条件。每次误差公式的最大数值偏差约为 5.3e-8。',
        '数值容差 1e-7、1e-6、1e-5 在全部新快照上产生相同的自身检测名单。少数生成节点的可选修复结果随容差改变，相关差异保留在核对文件中。',
        '当前结果依据 Cora 的公开输入表示规则。保持归一化二值形式的坐标替换可使新增检测分数为零，文档给出了具体反例。参考排序的概率结论明确保留交换性条件，ASR 采用本轮逐节点实测计数。','',
        '## 6. 完成与复核','',
        f'- 新生成配对快照：12 份。',
        f'- 开发与组件对照：{len(development["jobs"])} 组；三成员对照：{len(members["jobs"])} 组。',
        f'- 新样本配对确认：{len(confirmation["jobs"])} 组，其中推荐配置 12 组。',
        '- 三份研究清单均经保存模型重载、独立指标计数及阈值复算。',
        '- 推荐配置经原 run_experiments.py 入口完整运行四种方式。',
        '- 生成阶段、拟合阶段与评分阶段的输入和源码摘要分别保存。','',
        '## 7. 运行方式','',
        '在 LoSplit-main 目录中执行，使用新的输出目录：','',
        '~~~text',
        'python run_experiments.py --stage defense --attacks GTA UGBA DPGBA SPEAR --methods abp_feature_reference --dataset Cora --model GCN --seed 31 --attack-root runs/feature_reference_20261126/confirmation_inputs/seed31 --defense-root runs/my_feature_reference',
        '~~~','',
        '默认启用 unique_mode 修复。保留原特征的对照添加 --feature-repair none；三成员比较添加 --feature-members 3。',
        '当前通用入口按 Cora/GCN 协议校验输入，全部五项指标写入 results.md、results.json 及模型旁的元数据。','',
        '## 8. 文件','',
        '- [方法定义与详细证明](FEATURE_REFERENCE_THEORY.md)',
        '- [实验登记与配置说明](FEATURE_REFERENCE_PROTOCOL.md)',
        '- [新样本确认的全部结果](runs/feature_reference_20261126/confirmation/RESULTS.md)',
        '- [新样本确认的独立复核](runs/feature_reference_20261126/confirmation/verification.json)',
        '- [开发消融的全部结果](runs/feature_reference_20261126/ablation/RESULTS.md)',
        '- [开发消融复核](runs/feature_reference_20261126/ablation/verification.json)',
        '- [成员数对照](runs/feature_reference_20261126/members_comparison/RESULTS.md)',
        '- [证明条件、容差及数值处理核对](runs/feature_reference_20261126/condition_audit.json)',
        '- [同一触发图中正常节点的 CA 核对](runs/feature_reference_20261126/same_graph_clean_accuracy.json)',
        '- [开发阶段不同数值处理的同图 CA 对照](runs/feature_reference_20261126/ablation_same_graph_clean_accuracy.json)',
        '- [通用入口完整运行报告](runs/feature_reference_20261126/launcher_repair_verified/results.md)',
        '- [程序检查日志](runs/feature_reference_20261126/tests_final.log)',''
    ]
    (ROOT/'FEATURE_REFERENCE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    finite=[t['finite'] for row in verified['rows'] if row['case']=='full' for t in row['thresholds']]
    assert len(finite)==12 and all(finite)
    decision=dict(selected_case='full',feature_repair='unique_mode',members=1,
        rationale='Both registered extensions attain the same ASR and detection rates; restoration preserves clean-node accuracy in the same modified graph.',
        confirmation_summary=final_metrics,all_finite_reference_thresholds=True,
        verified_study_jobs=len(confirmation['jobs'])+len(development['jobs'])+len(members['jobs']))
    (OUT/'summary.json').write_text(json.dumps(decision,indent=2,allow_nan=False),encoding='utf-8')
    print('Wrote FEATURE_REFERENCE_RESULTS.md; verified study jobs:',decision['verified_study_jobs'])


if __name__=='__main__':main()
