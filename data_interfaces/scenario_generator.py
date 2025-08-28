import numpy as np
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ScenarioSet:
    seeds: List[int]

class ScenarioGenerator:
    """Generates an explicit set of scenarios represented by RNG seeds.
    Using the same ScenarioSet across different priority rules implements
    Common Random Numbers (CRN) for fair comparisons.
    """
    def __init__(self, n: int, base_seed: Optional[int] = 12345) -> None:
        self.n = int(n)
        self.base_seed = int(base_seed) if base_seed is not None else 12345

    def generate(self) -> ScenarioSet:
        # Deterministic, reproducible list of seeds
        rng = np.random.default_rng(self.base_seed)
        seeds = rng.integers(0, 2**32 - 1, size=self.n, dtype=np.uint32).astype(int).tolist()
        return ScenarioSet(seeds=seeds)

    @staticmethod
    def save(path: str, scen: ScenarioSet) -> None:
        with open(path, 'w') as f:
            json.dump({"seeds": scen.seeds}, f)

    @staticmethod
    def load(path: str) -> ScenarioSet:
        with open(path, 'r') as f:
            data = json.load(f)
        return ScenarioSet(seeds=list(map(int, data["seeds"])))