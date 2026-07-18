
from __future__ import annotations
from pathlib import Path
import time, copy, multiprocessing as mp
import pandas as pd, numpy as np, torch
from types import SimpleNamespace

from exact_env_mutual import run_snapshot_exact_episode, env_cfg_for, _DummyPlanner
from exact_env_mutual import evaluate_exact
from mutual_foundation import MutualRadarNet, DEVICE
from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from repaired_campaign_tools import EDFPlanner, ESTPlanner
from train_exact_sequence_head import Args as RewardArgs

OUT=Path(r'C:\Users\yousi\Downloads\radar_outputs\parallel_exact_mcts')
OUT.mkdir(parents=True,exist_ok=True)
CKPT=r'C:\Users\yousi\Downloads\radar_outputs\exact_train_qw02_seeded_more\exact_mutual_latest.pt'

BASE=dict(
    mode='diagnose', seed=0, ckpt=CKPT, iterations=1, episodes_per_iter=1, windows=500,
    max_targets_per_episode=10000, rollouts=4, horizon_windows=1, expand_top_k=12,
    c_puct=1.25, epsilon=0.0, rollout_policy='edf', seed_rollout_policies='planner_edf,planner_est,edf,est,edge',
    clone_mode='snapshot', plan_mode='window', window_extract='best', allow_retrack_in_window=False,
    stateless_tree_context=False, head_mode='pq', q_utility_weight=0.2, leaf_value_mix=0.0,
    select_mode='q', policy_target='q_softmax', policy_tau=1.0, search_alg='puct',
    max_num_considered_actions=16, gumbel_scale=1.0, mctx_value_scale=1.0, mctx_maxvisit_init=50.0,
    eager_edge_depth=1, prior_uniform_mix=0.0, rollout_est_prob=0.5, prior_mode='factorized',
    direct_mode='prob', direct_threshold=0.0, direct_q_alpha=0.0, direct_q_beta=1.0,
    direct_allow_retrack=False, direct_stateless_context=False, direct_cache_encoder=False,
    skip_direct_eval=True, accept_gate=False, gate_min_delta=0.0, gate_windows=50,
    gate_initials='5,15', gate_rates='0,2', gate_seeds='1', d_model=96, nhead=4, nlayers=2,
    gamma=0.98, replay_size=10000, train_steps=1, batch_size=64, lr=1e-4,
    train_initials='5,15,50,100', train_rates='0,2', train_grid=False, add_prefix_targets=False,
    target_selected_action=False, eval_initials='5,15,50,100', eval_rates='0,2', eval_seeds='986',
    env_mode='searched_sector_frame', track_update_reward=0.30, track_loss_penalty=4.0,
    search_refresh_tracked=0, search_refresh_gain=0.0, search_debt_penalty_weight=0.0,
    sector_staleness_weight=0.0, searched_sector_reward_weight=0.10, search_frame_overdue_weight=0.05,
    search_frame_desired_ms=3000.0, search_frame_deadline_ms=4500.0, search_frame_drop_penalty=4.0,
    penalize_hidden_targets=1,
)

def make_args(rollouts=4, windows=500):
    d=BASE.copy(); d['rollouts']=rollouts; d['windows']=windows
    return SimpleNamespace(**d)

def load_model():
    torch.set_num_threads(1)
    model=MutualRadarNet(d_model=96,nhead=4,nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(CKPT,map_location=DEVICE),strict=False)
    model.eval()
    return model

def run_cell(task):
    init,rate,seed,rollouts,windows=task
    model=load_model(); args=make_args(rollouts,windows)
    t0=time.perf_counter(); df,_=run_snapshot_exact_episode(model,args,init,rate,seed,train=False); wall=(time.perf_counter()-t0)*1000.0/max(1,len(df))
    return dict(planner=f'ParallelCell_r{rollouts}',initial_targets=init,rate=rate,seed=seed,reward_per_200ms_eq=float(df.window_reward.mean()),total_reward=float(df.window_reward.sum()),mean_delay_active=float(df.mean_delay_active.mean()),search_fraction=float(df.search_fraction.iloc[-1]),mean_active_targets=float(df.active_targets.mean()),mean_tracked_targets=float(df.tracked_targets.mean()),mean_drop_pct_active=float(df.drop_pct_active.mean()),planning_ms_per_200ms_eq=wall)

def eval_parallel(rollouts=4, workers=4, windows=500):
    tasks=[(i,r,986,rollouts,windows) for i in [5,15,50,100] for r in [0.0,2.0]]
    t0=time.perf_counter()
    with mp.get_context('spawn').Pool(processes=workers) as pool:
        rows=pool.map(run_cell,tasks)
    wall_total=(time.perf_counter()-t0)*1000.0
    raw=pd.DataFrame(rows)
    # Add heuristics serial for same cells.
    hrows=[]
    for init,rate,seed,_,_ in tasks:
        for name,hp in [('EDF',EDFPlanner(MAXT)),('EST',ESTPlanner(MAXT))]:
            seedall(seed); ww,_=run_fixed(hp,name,init,MAXT,seed,windows,200,env_cfg_for(rate,make_args(rollouts,windows))); ss=summarize_window_df(ww,'fixed'); ss.update(planner=name,initial_targets=init,rate=rate,seed=seed); hrows.append(ss)
    raw=pd.concat([raw,pd.DataFrame(hrows)],ignore_index=True,sort=False)
    raw.to_csv(OUT/f'parallel_r{rollouts}_w{workers}_raw.csv',index=False)
    summ=raw.groupby('planner',as_index=False).agg(reward=('reward_per_200ms_eq','mean'),drop=('mean_drop_pct_active','mean'),tracked=('mean_tracked_targets','mean'),active=('mean_active_targets','mean'),delay=('mean_delay_active','mean'),search=('search_fraction','mean'),latency=('planning_ms_per_200ms_eq','mean')).sort_values('reward',ascending=False)
    summ['wall_total_ms']=wall_total
    summ.to_csv(OUT/f'parallel_r{rollouts}_w{workers}_summary.csv',index=False)
    heur=raw[raw.planner.isin(['EDF','EST'])].groupby(['initial_targets','rate'])['reward_per_200ms_eq'].max(); gaps=[]
    for _,r in raw.iterrows():
        h=heur.loc[(r.initial_targets,r.rate)]; gaps.append(dict(planner=r.planner,init=r.initial_targets,rate=r.rate,reward=r.reward_per_200ms_eq,gap=r.reward_per_200ms_eq-h,win=r.reward_per_200ms_eq>=h,latency=r.planning_ms_per_200ms_eq,drop=r.mean_drop_pct_active,tracked=r.mean_tracked_targets,active=r.mean_active_targets))
    g=pd.DataFrame(gaps); g.to_csv(OUT/f'parallel_r{rollouts}_w{workers}_gaps.csv',index=False)
    print(summ.to_string(index=False)); print(g.groupby('planner').agg(wins=('win','sum'),avg_gap=('gap','mean'),min_gap=('gap','min')).sort_values(['wins','avg_gap'],ascending=False).to_string())

if __name__=='__main__':
    eval_parallel(rollouts=4,workers=4,windows=500)
