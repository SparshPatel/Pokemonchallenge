import os, sys, importlib.util
# Load our submission agent exactly as Kaggle would
SUB = os.path.abspath("submission")
spec = importlib.util.spec_from_file_location("subm_main", os.path.join(SUB, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
our_agent = m.agent

import kaggle_environments as ke

def play(opp, n=4):
    wins = errs = 0
    for g in range(n):
        env = ke.make("cabt", debug=True)
        # alternate seating
        if g % 2 == 0:
            env.run([our_agent, opp])
            me = 0
        else:
            env.run([opp, our_agent])
            me = 1
        s = env.steps[-1]
        r = env.state[me]["reward"] if False else None
        rew = env.steps[-1][me].get("reward")
        # detect errors on our side
        estep = None
        for step in env.steps:
            if step[me].get("status") == "ERROR" or step[me].get("error"):
                estep = step[me].get("error", "ERROR-status")
                errs += 1
                break
        rewards = [env.steps[-1][0].get("reward"), env.steps[-1][1].get("reward")]
        won = rewards[me] is not None and rewards[me] == max([x for x in rewards if x is not None])
        # more robust: reward 1 = win
        r_me = rewards[me]
        r_op = rewards[1-me]
        result = "WIN" if (r_me is not None and r_op is not None and r_me > r_op) else ("DRAW" if r_me==r_op else "LOSS")
        if result == "WIN": wins += 1
        print(f"  game {g}: seat={me} rewards={rewards} -> {result}" + (f"  ERR={estep}" if estep else ""))
    print(f"=> vs {opp}: {wins}/{n} wins, {errs} errored games")

for opp in ["random", "first"]:
    print(f"\n=== OUR AGENT vs {opp} ===")
    play(opp, 4)
print("ALLDONE")
