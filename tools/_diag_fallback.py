import os, sys, importlib.util, json
from collections import Counter
SUB = os.path.abspath("submission")
sys.path.insert(0, SUB)
sys.path.insert(0, os.path.abspath("src"))

from agent.adapter import extract_select, is_deck_phase
spec = importlib.util.spec_from_file_location("subm_main", os.path.join(SUB, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

stats = Counter()
by_type = Counter()
fallback_by_type = Counter()
errs = []

def spy(obs):
    try:
        if is_deck_phase(obs):
            return m._get_deck()
        sel = extract_select(obs)
        if sel is None:
            return m._get_deck()
        st = getattr(sel, "type", "?")
        by_type[st]+=1
        try:
            choice = m._get_policy().choose(obs, sel)
        except Exception as e:
            errs.append(f"choose raised on type {st}: {e!r}")
            stats["choose_raised"]+=1
            return m._safe_fallback(obs)
        fb = m._safe_fallback(obs)
        if not m._valid(choice, sel):
            stats["invalid_choice"]+=1
            fallback_by_type[st]+=1
            return fb
        # is the choice identical to the trivial fallback? (i.e. we didn't really decide)
        if list(choice) == list(fb):
            stats["equals_fallback"]+=1
            fallback_by_type[st]+=1
        else:
            stats["real_decision"]+=1
        return choice
    except Exception as e:
        errs.append(f"spy outer: {e!r}")
        return m._safe_fallback(obs)

import kaggle_environments as ke
env = ke.make("cabt", debug=True)
env.run([spy, "random"])
print("SELECTS BY TYPE:", dict(by_type))
print("FALLBACK/TRIVIAL BY TYPE:", dict(fallback_by_type))
print("STATS:", dict(stats))
print("ERRS(first5):", errs[:5])
rewards=[env.steps[-1][0].get("reward"), env.steps[-1][1].get("reward")]
print("REWARDS(me=0):", rewards)
print("DONE_DIAG")
