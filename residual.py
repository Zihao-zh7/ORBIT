"""Training-only frozen feature normalization/PCA and scaled ridge fitting."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class FeatureTransform:
    mean: np.ndarray
    std: np.ndarray
    directions: np.ndarray

    @classmethod
    def fit(cls, representations, width=64, chunk_size=256):
        if width < 1 or chunk_size < 1:
            raise ValueError('Feature width and chunk size must be positive')
        shape = representations.shape
        total = int(np.prod(shape[:-1]))
        if total == 0:
            raise ValueError('Training representations must be nonempty')
        # Shift before accumulating the mean, then form centered covariance.
        # Raw E[x*x]-E[x]**2 loses small variance at large feature offsets.
        origin = np.asarray(representations[0], dtype=np.float64).reshape(-1, shape[-1])[0].copy()
        sums = np.zeros(shape[-1], dtype=np.float64)
        for start in range(0, shape[0], chunk_size):
            b = np.asarray(representations[start:start+chunk_size], dtype=np.float64).reshape(-1, shape[-1])
            if not np.isfinite(b).all():
                raise ValueError('Training representations must be finite')
            sums += (b-origin).sum(0)
        mean = origin+sums/total
        gram = np.zeros((shape[-1], shape[-1]), dtype=np.float64)
        for start in range(0, shape[0], chunk_size):
            b = np.asarray(representations[start:start+chunk_size], dtype=np.float64).reshape(-1, shape[-1])-mean
            gram += b.T@b
        std = np.sqrt(np.maximum(np.diag(gram)/total, 0)).clip(1e-6)
        gram = gram / std[:, None] / std[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        # Gram eigenvalues cannot resolve singular values at SVD's squared epsilon.
        rank = np.count_nonzero(eigenvalues > max(eigenvalues[-1], 1)*np.finfo(float).eps*shape[-1])
        r = min(width-1, int(rank))
        directions = eigenvectors[:, -r:][:, ::-1] if r else eigenvectors[:, :0]
        return cls(mean, std, directions)

    def transform(self, representations):
        b = np.asarray(representations, dtype=np.float64)
        reduced = ((b-self.mean)/self.std) @ self.directions
        return np.concatenate((np.ones((*b.shape[:-1], 1)), reduced), axis=-1)


def residual_scale(residual, valid):
    residual = np.asarray(residual, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    count = mask.sum(axis=0)
    sums = np.where(mask, residual**2, 0).sum(axis=0)
    return _scale_from_moments(sums, count)


def residual_scale_cached(targets, predictions, valid, chunk_size=128):
    """Training RMS from cache chunks without a full residual tensor copy."""
    if chunk_size < 1 or targets.ndim != 3 or targets.shape != predictions.shape or targets.shape != valid.shape:
        raise ValueError('Matching [time,node,lead] arrays and a positive chunk size required')
    sums = np.zeros(targets.shape[1:], dtype=np.float64)
    count = np.zeros(targets.shape[1:], dtype=np.int64)
    for start in range(0, len(targets), chunk_size):
        residual = (np.asarray(targets[start:start+chunk_size], dtype=np.float64)
                    - np.asarray(predictions[start:start+chunk_size], dtype=np.float64))
        mask = np.asarray(valid[start:start+chunk_size], dtype=bool) & np.isfinite(residual)
        residual = np.where(mask, residual, 0)
        sums += np.einsum('tnd,tnd->nd', residual, residual)
        count += mask.sum(axis=0)
    return _scale_from_moments(sums, count)


def _scale_from_moments(sums, count):
    pooled_count = count.sum(axis=0)
    if np.any(pooled_count == 0):
        raise ValueError('Each lead needs at least one valid training target')
    pooled = np.sqrt(sums.sum(axis=0)/pooled_count)
    scale = np.sqrt(sums/np.maximum(count, 1))
    scale = np.where(count > 0, scale, pooled[None, :])
    return scale.clip(1e-6)


def ridge_statistics(design, residual, valid, scale, chunk_size=128):
    f, d = design.shape[-1], residual.shape[-1]
    gram = np.zeros((d, f, f), dtype=np.float64)
    rhs = np.zeros((d, f), dtype=np.float64)
    counts = np.zeros(d, dtype=np.int64)
    for start in range(0, len(design), chunk_size):
        phi = np.asarray(design[start:start+chunk_size], dtype=np.float64)
        r = np.asarray(residual[start:start+chunk_size], dtype=np.float64)
        mask = np.asarray(valid[start:start+chunk_size], dtype=bool) & np.isfinite(r)
        for h in range(d):
            weights = np.broadcast_to(1/scale[:, h], mask[..., h].shape)[mask[..., h]]
            x = phi[mask[..., h]]*weights[:, None]
            y = r[..., h][mask[..., h]]*weights
            gram[h] += x.T@x
            rhs[h] += x.T@y
            counts[h] += len(y)
    return gram, rhs, counts


def solve_ridge_statistics(statistics, beta):
    gram, rhs, counts = statistics
    if beta <= 0 or np.any(counts == 0):
        raise ValueError('Positive ridge and valid training labels required')
    weights = []
    for h in range(len(counts)):
        matrix = gram[h]/counts[h] + beta*np.eye(gram.shape[-1])
        weights.append(cho_solve(cho_factor(matrix, lower=True), rhs[h]/counts[h]))
    return np.asarray(weights).T


def fit_ridge(design, residual, valid, scale, beta):
    return solve_ridge_statistics(ridge_statistics(design, residual, valid, scale), beta)


def make_histories(features, gamma):
    z = np.zeros(features.shape[1:], dtype=np.float64)
    result = np.empty(features.shape, dtype=np.float64)
    for t, h in enumerate(features):
        z = gamma*z+(1-gamma)*h
        result[t] = z
    return result
