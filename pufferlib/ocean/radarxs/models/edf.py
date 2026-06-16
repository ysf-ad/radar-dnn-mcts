import numpy as np
from pufferlib.ocean.radarxs.models.planner import Planner

class EDFPlanner(Planner):
    SEARCH_ACTION = 0

    def __init__(self, max_trackers=500):
        super().__init__(max_trackers)

    def plan(self, obs, budget_ms=200):
        """
        Generate an EDF schedule over:
        - track deadlines (`t_deadline`)
        - search macro-sector urgency

        Search is modeled at the actual action granularity: one search action
        refreshes one non-overlapping 2x2 macro-sector, not one individual cell.
        Args:
            obs: Observation dict.
            budget_ms: Time budget to fill (used to size the search buffer).
        """
        candidates = self.build_search_candidates(obs, budget_ms)
        
        # 2. Track Candidates
        time_metric = obs['t_deadline']
        active_mask = obs['active_mask']
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            candidates.append({
                'action': int(idx) + 1,
                'time': time_metric[idx]
            })
            
        # 3. Sort EST
        candidates.sort(key=lambda x: x['time'])
        
        # 4. Return full sorted schedule
        plan = [c['action'] for c in candidates]

        # Safety fallback
        if not plan:
            plan = [self.SEARCH_ACTION]
            
        return plan
