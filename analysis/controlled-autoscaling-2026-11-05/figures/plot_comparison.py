# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Regenerate: uv run --no-project analysis/controlled-autoscaling-2026-11-05/figures/plot_comparison.py"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from orx_figstyle import WIDE, PALETTE, save, use_style

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE.parent/'aggregated-results.json').read_text())
MODELS=[('qwen35-4b','Qwen3.5-4B'),('qwen35-9b','Qwen3.5-9B'),
        ('nemotron35-lightning','Nemotron Lightning')]
DEFAULTS=json.loads((HERE.parent.parent/'controlled-warm-throughput-2026-11-03/default-spindle-configs-v0.1.4.json').read_text())['configs']
CONCURRENCIES=[1,8,16,32,64]
COLORS={'tinker':PALETTE['orange'],'spindle':PALETTE['blue'],'spindle_auto':PALETTE['green']}


def draw(metric, stem, title, unit, footer):
    assert len(DATA)==90 and all(r['repeats']==2 for r in DATA)
    fig,axes=plt.subplots(2,3,figsize=(WIDE,5.8),sharey='row')
    fig.set_layout_engine('constrained',rect=(0,.12,1,.73))
    fig.suptitle(title,x=.025,y=.985,ha='left',fontsize=11.5,fontweight='bold')
    fig.legend([Patch(facecolor=COLORS[b],edgecolor='#444',linewidth=.3) for b in COLORS],
               ['Tinker','Spindle · fixed 1 worker','Spindle · default autoscaling'],loc='upper left',bbox_to_anchor=(.025,.955),
               ncol=3,fontsize=7.0,handlelength=1.3)
    for row, task in enumerate(['ifeval','math500']):
        row_data=[d for d in DATA if d['task']==task]
        maximum=max(d[metric+'_max'] for d in row_data)*1.22
        for col,(short,label) in enumerate(MODELS):
            ax=axes[row,col]
            ax.text(0,1.14,('IFEval' if task=='ifeval' else 'MATH-500')+' · '+label,
                    transform=ax.transAxes,fontsize=6.8,fontweight='bold')
            cfg=DEFAULTS[col]
            ax.text(0,1.045,f"1 {cfg['inference_gpu']} · {cfg['inference_cpu']} CPU · {cfg['inference_memory_mib']//1024} GiB / worker",
                    transform=ax.transAxes,fontsize=6.2,color='#555')
            for i,backend in enumerate(COLORS):
                values=[next(d for d in DATA if d['model_short']==short and d['task']==task
                             and d['backend']==backend and d['request_concurrency']==c) for c in CONCURRENCIES]
                y=np.array([d[metric] for d in values])
                error=np.array([[d[metric]-d[metric+'_min'] for d in values],
                                [d[metric+'_max']-d[metric] for d in values]])
                bars=ax.bar(np.arange(5)+(i-1)*.26,y,width=.24,color=COLORS[backend],
                            edgecolor='#444',linewidth=.3,yerr=error,
                            error_kw=dict(elinewidth=.55,capsize=1.5,capthick=.5))
                labels=[f'{v/1000:.1f}k' if metric=='output_tokens_per_second' and v>=1000
                        else (f'{v:.0f}' if metric=='output_tokens_per_second' else f'{v:.2f}') for v in y]
                ax.bar_label(bars,labels=labels,padding=2,fontsize=5.1,rotation=90)
            ax.set_ylim(0,maximum)
            target=cfg['inference_target_concurrency']
            assert target in CONCURRENCIES
            ax.set_xticks(range(5),[f'{c}*' if c==target else str(c) for c in CONCURRENCIES])
            for concurrency,tick in zip(CONCURRENCIES,ax.get_xticklabels()):
                if concurrency==target:
                    tick.set_fontweight('bold')
            ax.set_xlabel('Concurrent requests',fontsize=6.6)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=6)
            ax.grid(axis='x',visible=False)
            ax.grid(axis='y',color='#e5e7eb',linewidth=.4)
            ax.set_axisbelow(True)
            if col==0:
                ax.set_ylabel(unit,fontsize=6.8)
            if metric=='output_tokens_per_second':
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v/1000:g}k' if v else '0'))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v:g}'))
    fig.text(.025,.025,footer+'\n* Spindle default autoscaling target per worker; client concurrency was swept.',fontsize=6.1,va='bottom',linespacing=1.6)
    save(fig,str(HERE/stem),close=False)
    fig.savefig(HERE/(stem+'.png'),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    use_style()
    draw('output_tokens_per_second','throughput-three-series','Warm throughput at matched request concurrency',
         'Output tokens / second',
         '128 IFEval prompts; 64 MATH problems × 4 samples. Two-pass means; whiskers show range.\n'
         'Warm API batches; scale-up during sampling included. Rows use different scales.')
    draw('allocated_billed_cost_usd','cost-three-series','Batch cost at matched request concurrency',
         'Allocated billed cost per batch (USD)',
         '128 IFEval prompts; 64 MATH problems × 4 samples. Two-pass means; whiskers show range.\n'
         'Billing: tokens / GPU time (auto: 1-min). Autoscaling includes GPUs retained across phases.')
