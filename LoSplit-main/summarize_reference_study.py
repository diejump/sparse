"""Write the research results page from completed, independently checked runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_study(name):
    folder = ROOT/'runs'/name
    study = json.loads((folder/'study.json').read_text(encoding='utf-8'))
    verification = json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='verified'
    assert verification['count']==len(study['jobs'])
    assert all(job['status']=='completed' for job in study['jobs'])
    return study


def main():
    ablation = load_study('reference_ablation_20261124')
    backbones = load_study('reference_backbones_20261124')
    baseline = [j for j in ablation['jobs'] if j['case']=='base'] + backbones['jobs']
    lines = [
        '# 参考判别版：本轮实测结果', '',
        '日期：2026-11-24。数据集为 Cora，图生成种子为 16，每个完整成员训练 200 轮。',
        '普通模型与参考模型分别使用 10、20、30 三个初始化种子；成员数消融按登记配置改变。',
        '这些设置用于当前开发分析。跨数据集与跨划分确认按预先保存的清单继续执行。', '',
        '## 1. 完成情况', '',
        f'- 四个方法参数、参考人数与抽样、三个组件：{len(ablation["jobs"])} 组完整训练。',
        f'- GAT、GraphSAGE：{len(backbones["jobs"])} 组完整训练。',
        '- 每组模型均保存后重新加载；独立核对脚本重新计算五项指标、整数计数及参考阈值。',
        '- 主启动器的 GCN 完整训练及 GraphSAGE 短训练通过 GPU 严格确定性入口；本机 GAT 的严格模式按已记录规则使用 CPU。',
        '- 程序检查覆盖指标、数据划分、参考标签、结构规则、三种模型与启动器。',
        '', '## 2. 三种模型的五项指标', '',
        '比例单位为 %。每行使用同一生成方式的保存快照。', '',
        '| 模型 | 方式 | CA | ASR | Precision | Recall | FPR |',
        '|---|---|---:|---:|---:|---:|---:|'
    ]
    for job in baseline:
        values = [f'{job["metrics"][k]*100:.4f}' if job['metrics'][k] is not None else '—'
                  for k in ('CA','ASR','Precision','Recall','FPR')]
        lines.append('| '+' | '.join([job['model'],job['kind'],*values])+' |')
    lines += ['', '## 3. 参数和组件观察', '',
        '| 方式 | 设置 | CA (%) | ASR (%) | 实际模型数 |',
        '|---|---|---:|---:|---:|']
    selected = ['base','ratio_0.1','ratio_0.3','count_120','risk_0.001','risk_0.009','risk_0.02',
                'members_1','members_5','reference_seed_20261125','no_reference','full_projection','keep_detected_labels']
    for kind in ('GTA','UGBA','DPGBA'):
        for case in selected:
            job = next(j for j in ablation['jobs'] if j['kind']==kind and j['case']==case)
            m=job['metrics']
            lines.append(f'| {kind} | {case} | {100*m["CA"]:.4f} | {100*m["ASR"]:.4f} | {m["ModelCount"]} |')
    lines += [
        '', '设置名称对应：ratio 为训练留出比例，count 为固定人数，risk 为排序参数，members 为每类模型成员数。',
        'no_reference 表示只使用普通模型；full_projection 表示完整 ABP 投影；keep_detected_labels 表示训练保留检出节点的标签。', '',
        '本轮比例 0.1、0.2、0.3 分别留出 55、109、163 个训练节点。固定 120 人作为历史设置对照。',
        '参考数目较少时，0.001 对应无限阈值，检出节点对推断类别的接受次数为零；元数据明确记录这一状态。',
        '成员数为 1 时，方法只训练两个模型。本轮可直接比较它与六个、十个模型的结果。',
        '组件贡献按本表中的配对变化判断。对于结果相同的组件设置，当前图设置提供的区分证据有限。',
        '并发运行时间保留为执行记录；模型效率比较应采用同设备串行重复计时。', '',
        '## 4. 参考标签与模型数量', '',
        '参考集合由留出的训练节点与移除训练身份重叠后的验证节点组成。参考标签仅在参考分类模型训练完成后用于阈值计算。',
        '普通分类模型按其独立规则使用训练和验证标签。默认完整方法共六个模型，其中三个参考分类模型承担额外的局部判别。',
        '拟合函数与评分函数分开，测试真实标签和真实目标名单进入最终评分。', '',
        '## 5. 下一阶段的确认实验', '',
        '已保存 Cora、Citeseer、Pubmed × GCN、GAT、GraphSAGE 的基础配置确认清单，包含新图种子和新模型种子组。',
        '这些条目目前处于计划状态。正式运行需要对应数据集与图设置的配对快照。',
        '当前开发结果用于选择进一步检验的配置；文章的跨数据集结论由后续实测建立。', '',
        '## 6. 文件', '',
        '- [完整参数与组件结果](runs/reference_ablation_20261124/RESULTS.md)',
        '- [参数与组件实验清单](runs/reference_ablation_20261124/study.json)',
        '- [参数与组件独立复核](runs/reference_ablation_20261124/verification.json)',
        '- [GAT、GraphSAGE 结果](runs/reference_backbones_20261124/RESULTS.md)',
        '- [模型结果独立复核](runs/reference_backbones_20261124/verification.json)',
        '- [跨数据集确认清单](runs/reference_confirmation_plan_20261124/study.json)',
        '- [主启动器完整训练报告](runs/reference_launcher_20261124_v3/results.md)',
        '- [程序检查日志](runs/reference_unit_tests_final.log)',
        '- [研究方案和完整推导](REFERENCE_METHOD.md)',
        '- [数据使用与论文对照](REFERENCE_AUDIT.md)',
        ''
    ]
    (ROOT/'REFERENCE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote REFERENCE_RESULTS.md with {len(ablation["jobs"])+len(backbones["jobs"])} verified study jobs')


if __name__=='__main__':main()
