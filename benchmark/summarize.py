"""Summarise results/raw.jsonl: Tables of the CityLearn-UA benchmark."""
import json,sys,collections,statistics as st
from pathlib import Path
R=[json.loads(l) for l in open(Path(__file__).parent/'results'/'raw.jsonl')]
C=[r for r in R if r['kind']=='cell'];O=[r for r in R if r['kind']=='outage']
cells=[("three_zone","summer"),("three_zone","autumn"),("three_zone","winter"),("dam","summer"),("dam","autumn"),("dam","winter"),("two_zone","summer")]
def g(rows,**k): return [r for r in rows if all(r.get(a)==b for a,b in k.items())]
def ms(a): return f"{st.mean(a):.3f}±{st.stdev(a):.3f} [{min(a):.3f},{max(a):.3f}]"
print('== score by cell')
for s,w in cells:
    row=[f"{s}/{w}"]
    for c in ('rbc','rbc_ua','sac'): row.append(f"{c} {g(C,signal=s,window=w,ctrl=c)[0]['score']:.3f}")
    for S in (3,5): row.append(f"mpcS{S} "+ms([r['score'] for r in g(C,signal=s,window=w,ctrl='mpc',S=S)]))
    print(' | '.join(row))
print('== money (no control; change; expensive)')
for s,w in cells:
    n=g(C,signal=s,window=w,ctrl='none')[0]
    row=[f"{s}/{w} none {n['cost_uah']:.0f}"]
    for c in ('rbc','rbc_ua'):
        r=g(C,signal=s,window=w,ctrl=c)[0];row.append(f"{c} {r['cost_uah']:.0f} {r['cost_change_pct']:+.1f}% exp {r.get('expensive_import_change_pct',float('nan')):+.1f}%")
    m=g(C,signal=s,window=w,ctrl='mpc',S=3)
    row.append("mpc "+f"{st.mean(x['cost_uah'] for x in m):.0f} "+ms([x['cost_change_pct'] for x in m])+(" exp "+ms([x['expensive_import_change_pct'] for x in m]) if 'expensive_import_change_pct' in m[0] else ''))
    print(' | '.join(row))
print('== peak components, summer')
for s in ('three_zone','dam'):
    for c in ('rbc','rbc_ua'):
        r=g(C,signal=s,window='summer',ctrl=c)[0];print(s,c,'all_time_peak',round(r['all_time_peak'],3))
    print(s,'mpc',ms([r['all_time_peak'] for r in g(C,signal=s,window='summer',ctrl='mpc',S=3)]))
print('== outage')
for sh in range(8):
    row=[f"shift {sh}"]
    for c in ('none','rbc','rbc_ua'):
        r=g(O,shift=sh,ctrl=c)[0];row.append(f"{c} U {r['unserved']:.3f} soc {r['soc_before_outage']:.3f}")
    m=g(O,shift=sh,ctrl='mpc');row.append("mpc U "+ms([x['unserved'] for x in m])+" soc "+ms([x['soc_before_outage'] for x in m]))
    print(' | '.join(row))
for c in ('none','rbc','rbc_ua','mpc'):
    per=[st.mean(x['unserved'] for x in g(O,shift=s,ctrl=c)) for s in range(8)]
    soc=[st.mean(x['soc_before_outage'] for x in g(O,shift=s,ctrl=c)) for s in range(8)]
    print(c,f"8 shifts U {min(per):.3f}-{max(per):.3f} mean {st.mean(per):.3f}; soc mean {st.mean(soc):.3f}")
w=sum(st.mean(x['unserved'] for x in g(O,shift=s,ctrl='mpc'))<g(O,shift=s,ctrl='rbc_ua')[0]['unserved'] for s in range(8));print('mpc better in',w,'of 8 (mean over seeds)')
print('per-seed wins',collections.Counter(sum(next(x for x in g(O,shift=s,ctrl='mpc') if x['noise']==n)['unserved']<g(O,shift=s,ctrl='rbc_ua')[0]['unserved'] for s in range(8)) for n in range(1,11)))
print('solver s',ms([r['solver_seconds_mean'] for r in C if r['ctrl']=='mpc' and r['S']==3]))
