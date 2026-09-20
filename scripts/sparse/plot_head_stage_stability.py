#!/usr/bin/env python3
"""Export cross-input stability figures from the completed M1 classification."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('classification',type=Path)
    parser.add_argument('--out-prefix',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.classification.read_text())
    if report['status']!='EXPLORATORY_CLASSIFICATION' or report['units']!=2160:
        raise ValueError('require the completed 2160-unit classification')
    entries=report['entries']
    fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
    colors=['#4477aa','#cc6677','#228833']
    for col,metric in enumerate(['action','video']):
        x=np.asarray([e[f'calibration_{metric}_relative_l2'] for e in entries])
        y=np.asarray([e[f'check_{metric}_relative_l2'] for e in entries])
        positive=np.concatenate([x[x>0],y[y>0]])
        floor=float(positive.min())/10 if len(positive) else 1e-12
        lx,ly=np.log10(np.maximum(x,floor)),np.log10(np.maximum(y,floor))
        for stage in range(3):
            selected=np.asarray([e['stage']==stage for e in entries])
            axes[0,col].scatter(lx[selected],ly[selected],s=10,alpha=.45,color=colors[stage],
                label=f'Stage {stage}',rasterized=True)
        bounds=[min(lx.min(),ly.min()),max(lx.max(),ly.max())]
        axes[0,col].plot(bounds,bounds,color='#333333',lw=1,linestyle='--')
        axes[0,col].set_xlabel(f'Calibration {metric}: log10 relative L2')
        axes[0,col].set_ylabel(f'Check input {metric}: log10 relative L2')
        rho=report[f'{metric}_rank_correlation']
        axes[0,col].set_title(f'{metric.capitalize()} sensitivity: Spearman ρ = {rho:.3f}' if rho is not None else f'{metric.capitalize()}: constant ranks')
        axes[0,col].legend(frameon=False,fontsize=9)
        if np.any(x==0) or np.any(y==0):axes[0,col].text(.03,.03,f'Zeros shown at {floor:.1e}',transform=axes[0,col].transAxes,fontsize=8)
    labels=['low_impact','action_sensitive','video_sensitive','mixed_sensitive','intermediate']
    display=['Low impact','Action','Video','Mixed','Intermediate']
    confusion=np.asarray([[report['confusion'][a][b] for b in labels] for a in labels])
    denominators=confusion.sum(axis=1,keepdims=True)
    fractions=confusion/np.maximum(denominators,1)
    shown=axes[1,0].imshow(fractions,cmap='Blues',vmin=0,vmax=1)
    axes[1,0].set_xticks(range(5),display,rotation=20,ha='right')
    axes[1,0].set_yticks(range(5),display)
    axes[1,0].set_xlabel('Type on the check input (same thresholds)')
    axes[1,0].set_ylabel('Type fitted on two inputs')
    for i in range(5):
        for j in range(5):
            axes[1,0].text(j,i,f'{confusion[i,j]}\n{fractions[i,j]:.0%}',ha='center',va='center',
                fontsize=9,color='white' if fractions[i,j]>.55 else '#222222')
    axes[1,0].set_title(f'Type agreement = {report["check_label_agreement"]:.1%}; counts / row %')
    fig.colorbar(shown,ax=axes[1,0],shrink=.8,label='Fraction within fitted type')
    x=np.arange(3)
    for offset,metric,color in [(-.18,'action','#cc503e'),(.18,'video','#4477aa')]:
        values=[s[f'{metric}_rank_correlation'] for s in report['per_stage']]
        axes[1,1].bar(x+offset,[v if v is not None else np.nan for v in values],width=.36,color=color,label=metric)
    axes[1,1].set_xticks(x,['Stage 0: steps 0–2','Stage 1: steps 3–6','Stage 2: steps 7–9'])
    axes[1,1].set_ylim(-1,1)
    axes[1,1].axhline(0,color='#999999',lw=.8)
    axes[1,1].set_ylabel('Spearman ρ: calibration vs check input')
    axes[1,1].set_title('Rank stability within each stage (720 units)')
    axes[1,1].legend(frameon=False)
    fig.suptitle('M1 cross-input check — one exposed Spatial initial observation; no SR claim',fontsize=14)
    for suffix in ['.png','.pdf']:fig.savefig(str(args.out_prefix)+suffix,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
