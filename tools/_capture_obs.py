import os, sys, importlib.util, json
SUB = os.path.abspath("submission")
sys.path.insert(0, SUB)
sys.path.insert(0, os.path.abspath("src"))

captured = []
# Monkeypatch: wrap the agent to capture raw obs on action steps
spec = importlib.util.spec_from_file_location("subm_main", os.path.join(SUB, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
real_agent = m.agent

def spy(obs):
    try:
        sel = obs.get("select") if isinstance(obs, dict) else None
        if sel is not None and len(captured) < 3:
            captured.append(json.loads(json.dumps(sel, default=str)))
    except Exception as e:
        captured.append({"spy_err": str(e)})
    return real_agent(obs)

import kaggle_environments as ke
env = ke.make("cabt", debug=True)
env.run([spy, "random"])
print("CAPTURED action-step select payloads (first 3):")
for i, c in enumerate(captured):
    print(f"--- select {i} ---")
    print(json.dumps(c, indent=1)[:1500])
print("DONE_CAPTURE")
