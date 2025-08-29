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
    
class ScenarioReducer:
    """
    Reduces a set of scenario embeddings into a representative subset using k-medoids clustering.
    Returns indices of selected medoids, their associated probability weights, and cluster assignments.
    """
    def __init__(self, k: int, rng_seed: int | None = 0):
        """
        Parameters:
        k (int): Number of medoids (representative scenarios) to select.
        rng_seed (int | None): Random seed for reproducibility of clustering.
        """
        self.k = int(k)
        self.rng_seed = rng_seed

    def standardize(self,X):
        """
        Standardizes features to have zero mean and unit variance.
        Avoids division by zero by setting std to 1 where std=0.

        Parameters:
        X (array-like): Input data matrix.

        Returns:
        np.ndarray: Standardized data matrix.
        """
        X = np.asarray(X, float)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0.0] = 1.0
        return (X - mu) / sd

    def pairwise_dist2(self,A, B):
        """
        Computes pairwise Euclidean distances between two sets of points.

        Parameters:
        A (np.ndarray): Array of shape (n,d).
        B (np.ndarray): Array of shape (k,d).

        Returns:
        np.ndarray: Distance matrix of shape (n,k) with distances ||a-b||_2.
        """
        # ||a-b||_2 pairwise, A:(n,d), B:(k,d) -> (n,k)
        A2 = np.sum(A*A, axis=1, keepdims=True)
        B2 = np.sum(B*B, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(A2 + B2 - 2*A@B.T, 0.0))

    def kmedoids(self,C, iters=20,rng=None):
        """
        Runs k-medoids clustering on a dissimilarity matrix C.

        Parameters:
        C (np.ndarray): Dissimilarity matrix of shape (n,n).
        iters (int): Maximum number of iterations.
        rng (int | None): Random seed or None for random initialization.

        Returns:
        tuple: (medoid indices (np.ndarray), cluster assignments (np.ndarray))
        """
        n = C.shape[0]
        rng = np.random.default_rng(None if rng is None else rng)
        medoids = rng.choice(n, size=self.k, replace=False)
        for _ in range(iters):
            # assign
            assign = np.argmin(C[:, medoids], axis=1)
            # update
            new_medoids = medoids.copy()
            for j in range(self.k):
                idx = np.where(assign == j)[0]
                if len(idx) == 0:  # empty cluster → re-seed
                    cand = rng.integers(0, n)
                    new_medoids[j] = cand
                else:
                    sub = C[np.ix_(idx, idx)]
                    new_medoids[j] = idx[np.argmin(sub.sum(axis=1))]
            if np.all(new_medoids == medoids):
                break
            medoids = new_medoids
        assign = np.argmin(C[:, medoids], axis=1)
        return medoids, assign

    def reduce(self, Xi: np.ndarray):
        """
        Reduces scenario embeddings by selecting representative medoids.

        Parameters:
        Xi (np.ndarray): Matrix of scenario embeddings of shape (m,d).

        Returns:
        tuple:
          keep_idx (np.ndarray): Indices of selected medoids in 0..m-1.
          weights (np.ndarray): Probability weights of clusters summing to 1, ordered by descending cluster size.
          assign (np.ndarray): Cluster assignment of each scenario to {0..k-1}.
        """
        Xs = self.standardize(np.asarray(Xi, float))
        C = self.pairwise_dist2(Xs, Xs)  # (m,m)
        medoids, assign = self.kmedoids(C, self.k, rng=self.rng_seed)
        # weights by cluster mass
        m = len(Xi)
        weights = np.zeros(self.k, float)
        for j in range(self.k):
            weights[j] = np.mean(assign == j)
        # order medoids by weight descending (nice to have)
        order = np.argsort(-weights)
        keep_idx = medoids[order]
        weights = weights[order]
        # remap assignments to new order
        remap = {old: new for new, old in enumerate(order)}
        assign = np.array([remap[a] for a in assign], int)
        return keep_idx, weights, assign