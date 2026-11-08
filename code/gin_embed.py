"""Pretrained GIN embeddings for a TDC ADMET task, written into the feature cache.

Run this in its own virtual environment, before the model run:

    DGLBACKEND=pytorch ../.venv-gin/bin/python gin_embed.py vdss_lombardo

It loads the supervised-masking GIN published with dgl-lifesci, embeds every unique
molecule of one task, takes the mean over atoms, and writes the result to the same
`.npz` cache file that `features.compute_block("gin", ...)` reads. The model run then
never imports torch: it only reads numbers.

Why a separate environment. dgl 2.2.0 ships its graphbolt library compiled against
torch 2.1 to 2.3 only, torchdata's datapipes were removed after 0.9, and that torch
needs numpy 1.x while the rest of this project is on numpy 2.5 and scipy 1.18. Those
constraints cannot be met in one environment, and they do not have to be: an embedding
is a fixed function of the weights and the molecule, so where it is computed makes no
difference to the result. The weights are downloaded as data into their own directory.

No third-party model code is run beyond the dgl-lifesci library itself, which is used
as a library, exactly as rdkit and lightgbm are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("DGLBACKEND", "pytorch")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)

MODEL_NAME = "gin_supervised_masking"
READOUT = "mean over atoms"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def unique_smiles(task: str, data_dir: str) -> list[str]:
    """Exactly the list protocol.TaskData builds, so the cache key matches."""
    from tdc.benchmark_group import admet_group

    group = admet_group(path=data_dir)
    bench = group.get(task)
    all_smiles = list(bench["train_val"]["Drug"]) + list(bench["test"]["Drug"])
    return sorted(set(all_smiles))


def embed(smiles: list[str], batch_size: int) -> np.ndarray:
    import dgl
    import torch
    from dgllife.model import load_pretrained
    from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(int(os.environ.get("GIN_THREADS", "4")))
    torch.set_grad_enabled(False)

    model = load_pretrained(MODEL_NAME)
    model.eval()
    atom_f, bond_f = PretrainAtomFeaturizer(), PretrainBondFeaturizer()

    out = np.zeros((len(smiles), 300), dtype=np.float32)
    n_bad = 0
    graphs, rows = [], []

    def flush() -> None:
        if not graphs:
            return
        bg = dgl.batch(graphs)
        nfeats = [bg.ndata.pop("atomic_number"), bg.ndata.pop("chirality_type")]
        efeats = [bg.edata.pop("bond_type"), bg.edata.pop("bond_direction_type")]
        node_repr = model(bg, nfeats, efeats)
        bg.ndata["h"] = node_repr
        pooled = dgl.mean_nodes(bg, "h").numpy()
        for i, r in enumerate(rows):
            out[r] = pooled[i]
        graphs.clear()
        rows.clear()

    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            n_bad += 1
            continue
        g = mol_to_bigraph(mol, add_self_loop=True, node_featurizer=atom_f,
                           edge_featurizer=bond_f, canonical_atom_order=False)
        if g is None or g.num_nodes() == 0:
            n_bad += 1
            continue
        graphs.append(g)
        rows.append(idx)
        if len(graphs) >= batch_size:
            flush()
            if (idx + 1) % (batch_size * 8) == 0:
                log(f"  {idx + 1}/{len(smiles)} molecules embedded")
    flush()
    log(f"  {len(smiles)} molecules, {n_bad} left as zeros (unparsable or empty)")
    return out


def host_reachable(host: str = "data.dgl.ai", port: int = 443, timeout: float = 8.0) -> str:
    """Empty string if the weights host answers a TLS handshake, else why not.

    Checked before anything expensive, because on this machine the handshake is
    refused outright and every attempt would otherwise pay for loading the splits and
    importing torch first.
    """
    import socket
    import ssl

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host):
                return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--data-dir", default=os.path.join(BASE, "data"))
    ap.add_argument("--cache-dir", default=os.path.join(BASE, "data", "_feature_cache"))
    ap.add_argument("--download-dir", default=os.path.join(BASE, "models", "dgllife"),
                    help="where the pretrained weights are kept, as data")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    os.environ["DGL_DOWNLOAD_DIR"] = args.download_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    import features  # after DGL_DOWNLOAD_DIR is set; features imports only rdkit

    weights = os.path.join(args.download_dir, f"{MODEL_NAME}.pth")
    if not os.path.exists(weights):
        why = host_reachable()
        if why:
            log(f"the pretrained weights are not in {args.download_dir} and data.dgl.ai is "
                f"unreachable from here ({why}). Nothing is written: a model run will use the "
                "candidates that need no embedding, and the result file records which ran.")
            return 2

    for task in args.tasks:
        smiles = unique_smiles(task, args.data_dir)
        path = features.cache_path("gin", smiles, args.cache_dir)
        if os.path.exists(path):
            log(f"{task}: already cached ({len(smiles)} molecules) -> {os.path.basename(path)}")
            continue
        log(f"{task}: embedding {len(smiles)} unique molecules with {MODEL_NAME} ({READOUT})")
        t0 = time.time()
        X = embed(smiles, args.batch_size)
        np.savez_compressed(path, X=X)
        log(f"{task}: wrote {os.path.basename(path)} shape {X.shape} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
