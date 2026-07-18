from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from adaptive_context_factorized import ContextPlanner, base_model, run_episode_with_feedback
from alpharadar_foundation_train import AlphaRadarNet, AlphaRadarMCTSPlanner
from edf_rank_v3_preserve_finetune import OUT as V3_OUT, V1, V2, env_for, load_ar
from final_radar_campaign import MAXT, build_env, get_obs, seedall, summarize_window_df
from hierarchical_sequence_transformer import tokenize, slot_features
from load_adaptive_train_eval import make_env
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS
from strict_window_report import execute_plan_until_budget

ROOT = Path(r"C:\Users\yousi\Downloads\Model1 1\CreateValid1\experiments\code")
OUT = ROOT / "CreateValid1" / "results" / "q_competition_mcts_20260518"
VISIBLE = Path(r"C:\Users\yousi\Downloads\radar_figures_visible")
OUT.mkdir(parents=True, exist_ok=True)
VISIBLE.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
V3 = V3_OUT / "edf_rank_v3_e10_kl0.1.pt"


def env_for_q(rate):
    env = make_env(float(rate))
    env["track_loss_penalty"] = 12.0
    env["track_urgency_bonus_weight"] = 1.0
    env["search_debt_penalty_weight"] = 0.001
    env["penalize_hidden_targets"] = 1
    return env


def load_model(path=V3):
    return load_ar(path)


class QCompetitionPlanner:
    """No scalar threshold: search and track compete on the same learned Q scale."""
    def __init__(self, model: AlphaRadarNet, mode="q", policy_weight=0.0, q_weight=1.0):
        self.model = model.eval()
        self.mode = str(mode)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def plan(self, obs, budget_ms=200):
        selected = set(); plan = []
        elapsed = 0.0; sc = tc = 0; last = -1
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        x0 = tokenize(self.adapt, obs, selected=None, search_count=0)
        tokens = torch.from_numpy(x0).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            active = tokens[:, :, 4] > 0.5; active[:, 0] = True
            emb = self.model.token_proj(tokens)
            cls = self.model.cls_token[None, None, :].expand(1, 1, -1)
            out = self.model.encoder(torch.cat([cls, emb], dim=1), src_key_padding_mask=~torch.cat([torch.ones((1,1), dtype=torch.bool, device=self.device), active], dim=1))
            cls_out = out[:, 0, :]; tok_out = out[:, 1:, :]
            cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
            active_np = active[0].detach().cpu().numpy().astype(bool)
            while elapsed < float(budget_ms) and len(plan) < 64:
                sf = slot_features(obs, elapsed, sc, tc, last, budget_ms)
                s = torch.from_numpy(sf).float().unsqueeze(0).to(self.device)
                slot_emb = self.model.slot_proj(s)
                type_ctx = torch.cat([cls_out, slot_emb], dim=-1)
                type_logit = self.model.type_head(type_ctx).squeeze(-1)[0]
                type_q = self.model.type_q_head(type_ctx)[0]
                slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
                track_ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
                track_logits = self.model.track_head(track_ctx).squeeze(-1)[0]
                track_q = self.model.track_q_head(track_ctx).squeeze(-1)[0]
                if self.mode == "policy":
                    search_score = float(F.logsigmoid(type_logit).detach().cpu())
                    track_scores = (F.logsigmoid(-type_logit) + F.log_softmax(track_logits, dim=0)).detach().cpu().numpy()
                elif self.mode == "policy_q":
                    search_score = float((self.policy_weight * F.logsigmoid(type_logit) + self.q_weight * type_q[1]).detach().cpu())
                    track_scores = (self.policy_weight * (F.logsigmoid(-type_logit) + F.log_softmax(track_logits, dim=0)) + self.q_weight * (type_q[0] + track_q)).detach().cpu().numpy()
                else:
                    search_score = float(type_q[1].detach().cpu())
                    track_scores = (type_q[0] + track_q).detach().cpu().numpy()
                mask = active_np.copy(); mask[0] = False
                deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
                live = np.zeros_like(mask); live[1:1+len(deadline)] = deadline > 0.0
                mask &= live
                for a in selected:
                    if 0 <= a < len(mask): mask[a] = False
                track_scores = np.where(mask, track_scores, -1e9)
                best = int(np.argmax(track_scores)); has = np.isfinite(track_scores[best]) and track_scores[best] > -1e8
                if (not has) or search_score >= float(track_scores[best]):
                    a = 0; elapsed += SEARCH_DWELL_MS; sc += 1
                else:
                    a = best; selected.add(a); elapsed += max(1.0, float(dwell[a-1]) if 1 <= a <= len(dwell) else SEARCH_DWELL_MS); tc += 1
                plan.append(int(a)); last = int(a)
        return plan or [0]


def collect_mcts(episodes=36, windows=60, rollouts=10):
    path = OUT / f"mcts_q_targets_e{episodes}_w{windows}_r{rollouts}.npz"
    if path.exists():
        z = np.load(path); return {k: z[k] for k in z.files}
    model = load_model().to(DEVICE).eval()
    cells = [("verylight", 8, 0.5), ("light", 15, 1.0), ("mid", 35, 2.0), ("main", 50, 2.0), ("heavy", 75, 5.0), ("stress", 100, 5.0)]
    xs=[]; slots=[]; pi=[]; q=[]; qmask=[]; returns=[]; meta=[]
    t0=time.perf_counter()
    for ep in range(episodes):
        label, init, rate = cells[ep % len(cells)]
        seed = 1010000 + ep; seedall(seed)
        env = env_for_q(rate)
        planner_cfg = dict(env)
        planner_cfg["planner_search_debt_penalty_weight"] = 0.001
        mcts = AlphaRadarMCTSPlanner(model, planner_cfg, rollouts=rollouts, c_puct=1.35, expand_top_k=14, training=True, prior_mode="factorized", belief_search_weight=0.35, belief_search_cap=8.0, q_scale=1.0)
        eng = build_env(mcts, init, MAXT, seed, 200, env); eng.reset(seed=seed); debt=0.0; traj=[]
        for w in range(windows):
            if eng.term_buf[0]: break
            obs = get_obs(eng, debt)
            plan, targets = mcts.plan_with_targets(obs, 200)
            reward, spent, debt, executed, _, _ = execute_plan_until_budget(eng, plan, 200.0, debt, "q_mcts_teacher", seed, w)
            per = float(reward) / max(1, len(targets))
            for tar in targets:
                tar.reward = per; traj.append(tar)
            if executed <= 0 or spent <= 0: break
        eng.close()
        G = 0.0
        for tar in reversed(traj):
            G = tar.reward + 0.985 * G
            xs.append(tar.x); slots.append(tar.slot); pi.append(tar.pi); q.append(tar.q); qmask.append(tar.q_mask); returns.append(G)
        row={"ep":ep,"label":label,"samples":len(xs),"elapsed_s":time.perf_counter()-t0}; meta.append(row); print("collect", row, flush=True)
    data={"x":np.asarray(xs,dtype=np.float32),"slot":np.asarray(slots,dtype=np.float32),"pi":np.asarray(pi,dtype=np.float32),"q":np.asarray(q,dtype=np.float32),"qmask":np.asarray(qmask,dtype=np.float32),"ret":np.asarray(returns,dtype=np.float32)}
    np.savez_compressed(path, **data); pd.DataFrame(meta).to_csv(OUT/"collect_meta.csv", index=False); return data


def train_q(data, epochs=8, kl_weight=0.08):
    ckpt = OUT / f"q_competition_e{epochs}_kl{kl_weight}.pt"
    if ckpt.exists(): return load_model(ckpt)
    m = load_model().to(DEVICE).train(); ref = load_model().to(DEVICE).eval()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=1e-4)
    x=torch.from_numpy(data["x"]).float().to(DEVICE); slot=torch.from_numpy(data["slot"]).float().to(DEVICE)
    pi=torch.from_numpy(data["pi"]).float().to(DEVICE); q=torch.from_numpy(data["q"]).float().to(DEVICE); qmask=torch.from_numpy(data["qmask"]).float().to(DEVICE); ret=torch.from_numpy(data["ret"]).float().to(DEVICE)
    scale=float(max(1.0, np.percentile(np.abs(np.concatenate([data["q"][data["qmask"]>0.5], data["ret"]])),90)))
    qn=q/scale; retn=ret/scale; n=x.shape[0]; logs=[]
    for ep in range(1, epochs+1):
        perm=torch.randperm(n,device=DEVICE); losses=[]; qlosses=[]; plosses=[]
        for st in range(0,n,384):
            idx=perm[st:st+384]
            tl,tr,v,tq,tqr=m(x[idx],slot[idx])
            with torch.no_grad(): rtl,rtr,_,rtq,rtqr=ref(x[idx],slot[idx])
            pi_s=pi[idx,0].clamp(0,1)
            type_loss=F.binary_cross_entropy_with_logits(tl,pi_s)
            mass=pi[idx,1:].sum(1); has=mass>1e-6
            if bool(torch.any(has)):
                target=pi[idx][has].clone(); target[:,0]=0; target=target/target.sum(1,keepdim=True).clamp_min(1e-6)
                rank_loss=F.kl_div(F.log_softmax(tr[has],dim=1),target,reduction="batchmean")
            else: rank_loss=torch.zeros((),device=DEVICE)
            search_q=qn[idx,0]
            track_q_target=(qn[idx,1:]*pi[idx,1:]).sum(1)/pi[idx,1:].sum(1).clamp_min(1e-6)
            type_q_target=torch.stack([track_q_target,search_q],dim=1)
            type_q_loss=F.smooth_l1_loss(tq,type_q_target)
            valid=qmask[idx]>0.5; valid[:,0]=False
            track_q_loss=F.smooth_l1_loss(tqr[valid],qn[idx][valid]) if bool(torch.any(valid)) else torch.zeros((),device=DEVICE)
            v_loss=F.smooth_l1_loss(v,retn[idx])
            kl_type=F.binary_cross_entropy_with_logits(tl,torch.sigmoid(rtl).detach())
            kl_track=F.kl_div(F.log_softmax(tr,dim=1),F.softmax(rtr,dim=1).detach(),reduction="batchmean")
            loss=0.4*type_loss+0.8*rank_loss+0.5*v_loss+1.2*type_q_loss+1.5*track_q_loss+kl_weight*(kl_type+kl_track)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            losses.append(float(loss.detach().cpu())); qlosses.append(float((type_q_loss+track_q_loss).detach().cpu())); plosses.append(float((type_loss+rank_loss).detach().cpu()))
        row={"epoch":ep,"loss":float(np.mean(losses)),"q_loss":float(np.mean(qlosses)),"policy_loss":float(np.mean(plosses)),"q_scale":scale}; logs.append(row); print("train",row,flush=True)
    m=m.cpu().eval(); torch.save(m.state_dict(),ckpt); pd.DataFrame(logs).to_csv(OUT/"train_log.csv",index=False); return m


def eval_all(model):
    v3=load_model(); v2=load_ar(V2); b=base_model(); ed=load_ar(ROOT/"CreateValid1/results/load_aware_edf_low_foundation_20260518/loadaware_e5_kl0.15.pt")
    scenarios=[("verylight8_r05",8,0.5,[100,110,120],260),("light15_r1",15,1.0,[100,110,120],260),("main50_r2",50,2.0,[100,110,120],260),("stress100_r5",100,5.0,[100,110,120],260),("heavy75_r5",75,5.0,[100,110,120],260)]
    specs={
        "QOnly":lambda init,env:QCompetitionPlanner(model,"q"),
        "PolicyQ":lambda init,env:QCompetitionPlanner(model,"policy_q",policy_weight=0.25,q_weight=1.0),
        "PolicyOnlyArgmax":lambda init,env:QCompetitionPlanner(model,"policy"),
        "V3_t0.39":lambda init,env:ContextPlanner.from_base(v3,0.39),
        "V3_t0.42":lambda init,env:ContextPlanner.from_base(v3,0.42),
        "V3_t0.44":lambda init,env:ContextPlanner.from_base(v3,0.44),
        "V3_t0.45":lambda init,env:ContextPlanner.from_base(v3,0.45),
        "V2_t0.45":lambda init,env:ContextPlanner.from_base(v2,0.45),
        "Base_t0.10":lambda init,env:ContextPlanner.from_base(b,0.10),
        "Base_t0.50":lambda init,env:ContextPlanner.from_base(b,0.50),
        "EDFLow_t0.35":lambda init,env:ContextPlanner.from_base(ed,0.35),
        "EDF":lambda init,env:EDFPlanner(MAXT),"EST":lambda init,env:ESTPlanner(MAXT)}
    raws=[]; wins=[]
    for label,init,rate,seeds,windows in scenarios:
        env=env_for_q(rate)
        for seed in seeds:
            for name,fac in specs.items():
                seedall(seed); w=run_episode_with_feedback(fac(init,env),name,init,seed,windows,env); s=summarize_window_df(w,"fixed"); s.update(scenario=label,planner=name,seed=seed,final_cumulative_reward=float(w["cumulative_reward"].iloc[-1])); raws.append(s); ww=w.copy(); ww["scenario"]=label; ww["planner"]=name; ww["seed"]=seed; wins.append(ww); print(label,seed,name,f"r={s['reward_per_200ms_eq']:.3f}",f"drop={s['mean_drop_pct_active']:.2f}",f"lat={s['planning_ms_per_200ms_eq']:.2f}",flush=True)
    raw=pd.DataFrame(raws); win=pd.concat(wins,ignore_index=True); raw.to_csv(OUT/"eval_raw.csv",index=False); win.to_csv(OUT/"eval_windows.csv",index=False)
    summary=raw.groupby(["scenario","planner"]).agg(reward=("reward_per_200ms_eq","mean"),cumulative=("final_cumulative_reward","mean"),drop=("mean_drop_pct_active","mean"),delay=("mean_delay_active","mean"),tracked=("mean_tracked_targets","mean"),active=("mean_active_targets","mean"),search=("search_fraction","mean"),latency=("planning_ms_per_200ms_eq","mean")).reset_index(); summary["score"]=summary["reward"]-3*summary["drop"]-0.002*summary["delay"]; summary=summary.sort_values(["scenario","score"],ascending=[True,False]); summary.to_csv(OUT/"eval_summary.csv",index=False); print(summary.to_string(index=False),flush=True)
    for scenario in summary.scenario.unique():
        keep=summary[summary.scenario.eq(scenario)].head(8).planner.tolist(); fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
        for planner in keep:
            sub=win[(win.scenario==scenario)&(win.planner==planner)]; by=sub.groupby("window"); t=(by["cumulative_reward"].mean().index+1)*0.2; axes[0,0].plot(t,by["cumulative_reward"].mean().values,label=planner,linewidth=1.8); axes[0,1].plot(t,by["window_reward"].mean().values,label=planner,linewidth=1.2); axes[1,0].plot(t,by["drop_pct_active"].mean().values,label=planner,linewidth=1.2); axes[1,1].plot(t,by["search_fraction"].mean().values,label=planner,linewidth=1.2)
        for ax,title in zip(axes.ravel(),["Cumulative reward","Window reward","Drop % active","Search fraction"]): ax.set_title(title); ax.set_xlabel("Time (s)"); ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7); p=OUT/f"{scenario}_qcomp_suite.png"; fig.savefig(p,dpi=180); fig.savefig(VISIBLE/p.name,dpi=180); plt.close(fig)


def main():
    t=time.perf_counter(); data=collect_mcts(); print("data",{k:v.shape for k,v in data.items()},flush=True); m=train_q(data); eval_all(m); print("elapsed",time.perf_counter()-t); print("OUT",OUT); print("VISIBLE",VISIBLE)

if __name__=="__main__": main()
