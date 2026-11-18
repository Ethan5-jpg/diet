"""Retain the pre-eligibility household role for the archived site-specific PCA."""
import numpy as np

from adapters.person_inputs import CONTRAST, PersonSource


class PCARoleSource(PersonSource):
    def split(self, train, valid, contrast=CONTRAST):
        # The inherited role builder checks train/validation household isolation,
        # applies the frozen CLR and outcome transforms, then returns eligible rows.
        split = super().split(train, valid, contrast)
        rows = np.flatnonzero(train)
        for site, transform in split.preprocessing['CLR'].items():
            if transform['training_participants'] != len(rows):
                raise ValueError(f'{site}: training-role size differs from CLR registration')
        rows.flags.writeable = False
        split.pca_training_rows = rows
        return split


def wrap_source(source):
    """Reuse the verified native source and its role/split methods without data copies."""
    if not isinstance(source, PersonSource):
        raise TypeError('Verified V6 PersonSource required for V7 native PCA replay')
    return PCARoleSource(source._data, source.extra, source.source_hashes, source.historical_scores)
