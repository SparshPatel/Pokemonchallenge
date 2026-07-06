import os, sys, importlib.util
SUB = os.path.abspath("submission")
spec = importlib.util.spec_from_file_location("subm_main", os.path.join(SUB, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
our_agent = m.agent
import kaggle_environments as ke

def play(opp, n):
    w=l=d=0
    for g in range(n):
        env = ke.make("cabt", debug=False)
        me = g % 2
        pair = [our_agent, opp] if me==0 else [opp, our_agent]
        env.run(pair)
        r=[env.steps[-1][0].get("reward"), env.steps[-1][1].get("reward")]
        rm,ro=r[me],r[1-me]
        if rm is None or ro is None: d+=1
        elif rm>ro: w+=1
        elif rm<ro: l+=1
        else: d+=1
    print(f"vs {opp}: {w}W-{l}L-{d}D / {n}  winrate={w/n:.2f}")
    return w,n

import sys as _s
N=int(_s.argv[1]) if len(_s.argv)>1 else 20
for opp in ["random","first"]:
    play(opp, N)
print("BENCH_DONE")
