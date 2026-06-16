from abc import ABC, abstractmethod
import numpy as np

class Planner(ABC):
    """
    Abstract Base Class for Radar Planners.
    """
    def __init__(self, max_trackers=500):
        self.max_trackers = max_trackers

    @staticmethod
    def macro_search_urgencies(grid):
        """Return sorted search urgencies at the actual search-action granularity.

        One search action refreshes one non-overlapping 2x2 macro-sector, so
        heuristic search scheduling should rank those macro-sectors rather than
        individual grid cells.

        We score each search action by the minimum remaining freshness across
        the 2x2 patch it would refresh. This keeps search expressed in the same
        timing language as EDF/EST: the patch whose worst sector will go stale
        first is the most urgent search candidate.
        """
        grid = np.asarray(grid, dtype=np.float32).reshape(10, 30)
        urgencies = []
        for r in range(0, 9, 2):
            for c in range(0, 29, 2):
                patch = np.asarray([
                    grid[r, c],
                    grid[r, c + 1],
                    grid[r + 1, c],
                    grid[r + 1, c + 1],
                ], dtype=np.float32)
                urgencies.append(float(np.min(patch)))
        return np.sort(np.asarray(urgencies, dtype=np.float32))

    @classmethod
    def build_search_candidates(cls, obs: dict, budget_ms: float) -> list:
        search_dwell = 10.0
        limit = int(budget_ms / search_dwell) + 2
        macro_urgencies = cls.macro_search_urgencies(obs["grid"])
        candidates = []
        for i in range(limit):
            urgency = float(macro_urgencies[min(i, len(macro_urgencies) - 1)])
            candidates.append({"action": 0, "time": urgency})
        return candidates

    @abstractmethod
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list:
        """
        Generate a sequence of actions given the observation and time budget.
        
        Args:
            obs: Dictionary containing 'grid', 'trackers', 't_dwell', etc.
                 - grid: (GRID_SIZE,) float array of staleness
                 - trackers: (MAX_TRACKERS, FEATURES) float array
                 - t_dwell: (MAX_TRACKERS,) float array of estimated dwell times
                 - priority: (MAX_TRACKERS,) float array (target priority)
                 - active_mask: (MAX_TRACKERS,) bool array
            budget_ms: Time budget in milliseconds for the window.
            
        Returns:
            List of integers representing action indices.
            0 = Search
            1..N = Track Target i-1
        """
        pass
