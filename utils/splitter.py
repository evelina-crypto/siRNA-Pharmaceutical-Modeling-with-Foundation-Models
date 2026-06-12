from sklearn.model_selection import GroupKFold
from sklearn.model_selection._split import _BaseKFold
import numpy as np

class GroupKFoldLeakPerGroup(_BaseKFold):
    def __init__(self, n_splits=5, leak_n=30, random_state=42):
        # shuffle and random state disabled with GroupK_fold
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.leak_n_per_group = leak_n
        self.random_state=random_state

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("You must pass `groups` for GroupKFoldLeakPerGroup")

        gkf = GroupKFold(n_splits=self.n_splits)
        rng = np.random.default_rng(self.random_state)
        groups = np.asarray(groups)

        for train_idx, test_idx in gkf.split(X, y, groups):
            # find all the unique gene‐groups in this test fold
            test_groups = np.unique(groups[test_idx])

            # collect leak indices group by group
            leak_indices = []
            for g in test_groups:
                mask = (groups[test_idx] == g)
                group_idx = test_idx[mask]

                n_leak = min(self.leak_n_per_group, len(group_idx))
                if n_leak == 0:
                    continue

                chosen = rng.choice(group_idx, size=n_leak, replace=False)
                leak_indices.append(chosen)

            if leak_indices:
                leak_idx = np.concatenate(leak_indices)

                # Remove leaked indices from the test set
                test_idx = np.setdiff1d(test_idx, leak_idx, assume_unique=True)

                # Add leaked indices to the training set
                train_idx = np.unique(np.concatenate([train_idx, leak_idx]))

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits