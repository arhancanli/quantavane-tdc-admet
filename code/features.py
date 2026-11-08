"""Molecular featurisation for the TDC ADMET benchmark group.

Every feature here is computed from one molecule at a time and uses no labels and
no information from any other molecule, so computing features over train+test
together cannot leak label information. Feature *selection* and all model fitting
happen strictly inside the training folds (see protocol.py).

Blocks
------
desc      ~210 RDKit 2D descriptors
morgan    Morgan (ECFP-like) count fingerprint, radius 2, 2048 columns
fcfp      Morgan feature-invariant count fingerprint, radius 3, 1024 columns
avalon    Avalon count fingerprint, 1024 columns
maccs     MACCS structural keys, 167 columns
atompair  Atom-pair count fingerprint, 2048 columns (added for round 2: pairs of
          atom types at a topological distance, which no Morgan radius captures)
rdkfp     RDKit path fingerprint, 2048 columns (added for round 2: linear and
          branched paths up to 7 bonds, a different substructure vocabulary again)
erg       Extended reduced graph fingerprint, 315 columns (round 3): pharmacophore
          node types at topological distances, a coarser and more shape-like view
          than any substructure count
gin       300 columns (round 3): the mean of the node embeddings from the
          pretrained supervised-masking GIN published with dgl-lifesci. This block
          is NOT computed here -- it needs torch and dgl, which do not coexist with
          this environment's versions. It is produced once by code/gin_embed.py in
          its own virtual environment and read from the same cache files as every
          other block, so a model run never imports torch.

Results are cached to .npz keyed by a hash of the exact SMILES list, so repeated
runs and multiple tasks never recompute the same block.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs

RDLogger.DisableLog("rdApp.*")

BLOCKS = ("desc", "morgan", "fcfp", "avalon", "maccs", "atompair", "rdkfp", "erg", "gin")
# Blocks a model run can compute for itself; "gin" is filled in beforehand.
LOCAL_BLOCKS = tuple(b for b in ("desc", "morgan", "fcfp", "avalon", "maccs",
                                 "atompair", "rdkfp", "erg"))
GIN_WIDTH = 300

_DESC_LIST = list(Descriptors._descList)
DESC_NAMES = [name for name, _ in _DESC_LIST]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FCFP = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=1024,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)
_ATOMPAIR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
_RDKFP = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7, fpSize=2048)

_WIDTHS = {"morgan": 2048, "fcfp": 1024, "avalon": 1024, "maccs": 167,
           "atompair": 2048, "rdkfp": 2048, "erg": 315, "gin": 300}

# RDKit descriptors occasionally return inf/nan (Ipc overflows on large graphs).
# Clip to a finite window instead of dropping the column.
_CLIP = 1e6


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def _desc_row(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(len(_DESC_LIST), dtype=np.float64)
    out = np.empty(len(_DESC_LIST), dtype=np.float64)
    for i, (_, fn) in enumerate(_DESC_LIST):
        try:
            out[i] = fn(mol)
        except Exception:
            out[i] = 0.0
    return out


def _block_for_mol(block: str, mol) -> np.ndarray:
    if block == "desc":
        return _desc_row(mol)
    if mol is None:
        return np.zeros(_WIDTHS[block], dtype=np.float64)
    if block == "morgan":
        return _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float64)
    if block == "fcfp":
        return _FCFP.GetCountFingerprintAsNumPy(mol).astype(np.float64)
    if block == "atompair":
        return _ATOMPAIR.GetCountFingerprintAsNumPy(mol).astype(np.float64)
    if block == "rdkfp":
        return _RDKFP.GetCountFingerprintAsNumPy(mol).astype(np.float64)
    if block == "erg":
        return np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float64)
    if block == "avalon":
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=1024)
        arr = np.zeros(1024, dtype=np.float64)
        for idx, val in fp.GetNonzeroElements().items():
            arr[idx] = val
        return arr
    if block == "maccs":
        bv = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(len(bv), dtype=np.float64)
        for idx in bv.GetOnBits():
            arr[idx] = 1.0
        return arr
    raise ValueError(f"unknown block {block!r}")


def _cache_key(block: str, smiles: list[str]) -> str:
    h = hashlib.sha256()
    h.update(block.encode())
    for s in smiles:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:20]


def cache_path(block: str, smiles: list[str], cache_dir: str) -> str:
    """Where one block's cache for exactly this SMILES list lives."""
    return os.path.join(cache_dir, f"{block}_{_cache_key(block, smiles)}.npz")


def compute_block(block: str, smiles: list[str], cache_dir: str | None = None) -> np.ndarray:
    """Return an (n_molecules, n_features) float32 matrix for one feature block."""
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = cache_path(block, smiles, cache_dir)
        if os.path.exists(path):
            return np.load(path)["X"]

    if block == "gin":
        # Never computed inside a model run: torch and dgl need older numpy and scipy
        # than everything else here. A missing cache is an error with instructions,
        # not a silent zero matrix, which would look like a feature block that simply
        # does not help.
        raise RuntimeError(
            "the 'gin' block is not cached for this SMILES list. Produce it first:\n"
            "  DGLBACKEND=pytorch ../.venv-gin/bin/python gin_embed.py <task> "
            f"--cache-dir {cache_dir}\n"
            f"expected file: {path}"
        )

    mols = [_mol(s) for s in smiles]
    rows = [_block_for_mol(block, m) for m in mols]
    X = np.vstack(rows)
    X[~np.isfinite(X)] = 0.0
    np.clip(X, -_CLIP, _CLIP, out=X)
    X = X.astype(np.float32)

    if path:
        np.savez_compressed(path, X=X)
    return X


def featurise(smiles: list[str], blocks, cache_dir: str | None = None) -> np.ndarray:
    """Horizontally stack the requested blocks for the given SMILES, in order."""
    mats = [compute_block(b, smiles, cache_dir) for b in blocks]
    return np.hstack(mats) if len(mats) > 1 else mats[0]


def n_invalid(smiles: list[str]) -> int:
    """How many SMILES RDKit could not parse (reported in the run log)."""
    return sum(1 for s in smiles if _mol(s) is None)
