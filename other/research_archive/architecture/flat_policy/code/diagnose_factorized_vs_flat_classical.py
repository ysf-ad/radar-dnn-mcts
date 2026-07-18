
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from final_radar_campaign import MAXT, build_env, get_obs, seedall
from repaired_campaign_tools import EDFPlanner, SEARCH_DWELL_MS, execute_first_valid_action
from mutual_alpha_radar_loop import configured_env
from mutual_foundation import MutualRadarNet, DEVICE, action_priors_from_logits
from mutual_features import tokenize, slot_features
from realistic_reward_retrain import adapter

OUT=Path(r"C:\Users\yousi\Downloads\radar_outputs\mcts_sched_v1_factorized_classical_train\diagnostics")
OUT.mkdir(parents=True, exist_ok=True)
CKPT=Path(r"C:\Users\yousi\Downloads\radar_outputs\mcts_sched_v1_factorized_classical_train\exact_mutual_latest.pt")

def args():
    return SimpleNamespace(
        env_mode='mcts_sched_v1', track_update_reward=0.65, track_loss_penalty=8.0,
        searched_sector_reward_weight=0.12, search_frame_overdue_weight=0.20,
        search_frame_desired_ms=1800.0, search_frame_deadline_ms=3600.0, search_frame_drop_penalty=8.0,
        search_refresh_tracked=0, search_refresh_gain=0.0, search_debt_penalty_weight=0.0,
        sector_staleness_weight=0.0, penalize_hidden_targets=1,
    )

def load_model():
    m=MutualRadarNet(d_model=96,nhead=4,nlayers=2).to(DEVICE)
    m.load_state_dict(torch.load(CKPT,map_location=DEVICE), strict=False)
    m.eval(); return m

def main():
    model=load_model(); adapt=adapter(); rows=[]
    A=args()
    for init in [5,15,50,75,100]:
      for rate in [0.0,2.0,3.0]:
        seed=1031; seedall(seed)
        env=configured_env(rate,A)
        eng=build_env(EDFPlanner(MAXT), init, MAXT, seed, 200, env)
        eng.reset(seed=seed)
        debt=0.0
        for w in range(10):
            obs=get_obs(eng,debt)
            x=tokenize(adapt, obs, selected=set(), search_count=0)
            slot=slot_features(obs,0.0,0,0,-1,200.0)
            with torch.no_grad():
                tl,tr,val,tq,trq=model(torch.from_numpy(x).float().unsqueeze(0).to(DEVICE), torch.from_numpy(slot).float().unsqueeze(0).to(DEVICE))
            pf=action_priors_from_logits(tl[0],tr[0],'factorized')
            pl=action_priors_from_logits(tl[0],tr[0],'flat')
            active=np.asarray(obs['active_mask']).astype(bool)
            tracked=active & (np.asarray(obs['t_deadline'],dtype=np.float32)>=0)
            valid_track=np.where(tracked)[0]+1
            topf=np.argsort(-pf)[:8]
            topl=np.argsort(-pl)[:8]
            rows.append(dict(
                init=init, rate=rate, window=w, active=int(active.sum()), tracked=int(tracked.sum()),
                type_logit=float(tl[0].detach().cpu()), factorized_p_search=float(pf[0]), flat_p_search=float(pl[0]),
                factorized_track_mass=float(pf[1:].sum()), flat_track_mass=float(pl[1:].sum()),
                max_track_prior_factorized=float(pf[valid_track].max()) if len(valid_track) else 0.0,
                max_track_prior_flat=float(pl[valid_track].max()) if len(valid_track) else 0.0,
                top1_factorized=int(topf[0]), top1_flat=int(topl[0]),
                top8_search_factorized=int(0 in topf), top8_search_flat=int(0 in topl),
                top8_tracks_factorized=int(sum(1 for a in topf if a!=0)), top8_tracks_flat=int(sum(1 for a in topl if a!=0)),
            ))
            # advance with EDF to sample realistic states
            plan=EDFPlanner(MAXT).plan(obs,200)
            elapsed=0.0
            for act in plan:
                ok,dt,newdebt=execute_first_valid_action(eng,int(act),debt)
                if not ok: continue
                elapsed += float(dt); debt=float(newdebt)
                if elapsed>=200.0: break
        eng.close()
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'prior_diagnostics.csv', index=False)
    summ=df.groupby(['init','rate']).agg(
        fact_search=('factorized_p_search','mean'), flat_search=('flat_p_search','mean'),
        fact_max_track=('max_track_prior_factorized','mean'), flat_max_track=('max_track_prior_flat','mean'),
        fact_top1_search=('top1_factorized', lambda s: float(np.mean(np.asarray(s)==0))),
        flat_top1_search=('top1_flat', lambda s: float(np.mean(np.asarray(s)==0))),
        tracked=('tracked','mean')
    ).reset_index()
    summ.to_csv(OUT/'prior_diagnostics_summary.csv', index=False)
    print(summ.to_string(index=False))
    print(OUT)
if __name__=='__main__': main()
