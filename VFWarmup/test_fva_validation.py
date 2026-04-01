"""
Validate VFWarmup FVA output against COBRApy FVA.

Downloads a model from BiGG, generates fixed-format MPS, runs VFWarmup,
then compares the first 2*nRxns warmup columns (FVA bounds) against COBRApy.

Usage:
    python test_fva_validation.py <bigg_model_id> <vfwarmup_binary> [tolerance]

Example:
    python test_fva_validation.py iJN746 ./createWarmupPtsGLPK
    python test_fva_validation.py iML1515 ./createWarmupPtsGLPK 1e-2
"""
import sys
import os
import subprocess
import tempfile
import gzip
import shutil
import numpy as np
import pandas as pd
from scipy import sparse


def download_model(model_id, out_dir):
    """Download SBML model from BiGG and return cobra model."""
    import cobra
    import httpx

    xml_path = os.path.join(out_dir, f"{model_id}.xml")
    gz_path = xml_path + ".gz"

    url = f"http://bigg.ucsd.edu/static/models/{model_id}.xml.gz"
    print(f"  Downloading {url} ...")
    client = httpx.Client(timeout=120.0)
    r = client.get(url)
    r.raise_for_status()
    with open(gz_path, 'wb') as f:
        f.write(r.content)

    with gzip.open(gz_path, 'rb') as f_in, open(xml_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    model = cobra.io.read_sbml_model(xml_path)
    print(f"  {model_id}: {len(model.reactions)} rxns, {len(model.metabolites)} mets")
    return model


def write_fixed_mps(model, mps_path):
    """Write fixed-format MPS from a COBRA model."""
    import cobra
    S = cobra.util.create_stoichiometric_matrix(model)
    nMets, nRxns = S.shape
    Scsc = sparse.csc_matrix(S)
    lbs = [r.lower_bound for r in model.reactions]
    ubs = [r.upper_bound for r in model.reactions]

    with open(mps_path, 'w') as f:
        f.write(f"{'NAME':<14}{model.id or 'model'}\n")
        f.write("ROWS\n")
        f.write(" N  COST\n")
        for i in range(nMets):
            f.write(f" E  EQ{i+1}\n")

        f.write("COLUMNS\n")
        for j in range(nRxns):
            col_start = Scsc.indptr[j]
            col_end = Scsc.indptr[j + 1]
            col_name = f"X{j+1}"
            for idx in range(col_start, col_end):
                row = Scsc.indices[idx]
                val = Scsc.data[idx]
                f.write(f"    {col_name:<10}{'EQ' + str(row+1):<10}{val:<12g}\n")

        f.write("RHS\n")
        f.write("BOUNDS\n")
        for j in range(nRxns):
            col_name = f"X{j+1}"
            lb, ub = lbs[j], ubs[j]
            if lb == 0 and ub == 0:
                f.write(f" FX BOUND     {col_name:<10}{0:<12g}\n")
            else:
                if lb != 0:
                    f.write(f" LO BOUND     {col_name:<10}{lb:<12g}\n")
                f.write(f" UP BOUND     {col_name:<10}{ub:<12g}\n")
        f.write("ENDATA\n")

    rxn_ids = [r.id for r in model.reactions]
    print(f"  MPS written: {nRxns} cols, {nMets} rows")
    return nRxns, rxn_ids


def run_vfwarmup(binary, mps_path, nRxns):
    """Run VFWarmup and return FVA bounds CSV path."""
    nPts = 2 * nRxns
    cmd = ["mpirun", "-np", "1", "--oversubscribe", "--bind-to", "none",
           binary, mps_path, str(nPts)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"VFWarmup failed with exit code {result.returncode}")
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"  {line.strip()}")

    base = os.path.splitext(mps_path)[0]
    fva_csv = f"{base}_fva_bounds.csv"
    if not os.path.exists(fva_csv):
        raise FileNotFoundError(f"Expected FVA bounds at {fva_csv}")
    return fva_csv


def run_cobra_fva(model):
    """Run COBRApy FVA and return DataFrame."""
    import cobra
    print("  Running COBRApy FVA (single process)...")
    fva = cobra.flux_analysis.flux_variability_analysis(
        model, fraction_of_optimum=0.0, processes=1
    )
    print(f"  COBRApy FVA: {len(fva)} reactions")
    return fva


def compare_fva(vf_csv, cobra_fva, rxn_ids, tol=1e-3):
    """Compare VFWarmup FVA bounds against COBRApy FVA. Returns True if all match."""
    vf = pd.read_csv(vf_csv)
    name_to_id = {f"X{j+1}": rid for j, rid in enumerate(rxn_ids)}

    matched = 0
    min_diffs = []
    max_diffs = []
    mismatches = []

    for _, row in vf.iterrows():
        rxn_id = name_to_id.get(row['reaction'])
        if rxn_id and rxn_id in cobra_fva.index:
            matched += 1
            vf_min, vf_max = row['min'], row['max']
            cb_min = cobra_fva.loc[rxn_id, 'minimum']
            cb_max = cobra_fva.loc[rxn_id, 'maximum']
            min_diffs.append(abs(vf_min - cb_min))
            max_diffs.append(abs(vf_max - cb_max))
            if abs(vf_min - cb_min) > tol or abs(vf_max - cb_max) > tol:
                mismatches.append((row['reaction'], rxn_id,
                                   vf_min, cb_min, vf_max, cb_max))

    min_diffs = np.array(min_diffs)
    max_diffs = np.array(max_diffs)

    print(f"\n=== FVA Comparison ({matched}/{len(vf)} matched) ===")
    print(f"  Min bounds - mean diff: {min_diffs.mean():.8f}, max: {min_diffs.max():.6f}")
    print(f"  Max bounds - mean diff: {max_diffs.mean():.8f}, max: {max_diffs.max():.6f}")
    print(f"  Match (tol={tol}): min={(min_diffs < tol).sum()}/{matched}, "
          f"max={(max_diffs < tol).sum()}/{matched}")

    if mismatches:
        print(f"\n  Mismatches: {len(mismatches)}")
        for m in mismatches[:10]:
            print(f"    {m[0]:<10} {m[1]:<20} VF=[{m[2]:.4f},{m[4]:.4f}] "
                  f"CB=[{m[3]:.4f},{m[5]:.4f}]")
        return False

    print("  PASSED - all FVA bounds match!")
    return True


if __name__ == '__main__':
    import cobra

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <bigg_model_id> <vfwarmup_binary> [tolerance]")
        sys.exit(1)

    model_id = sys.argv[1]
    binary = sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[1] Downloading {model_id} from BiGG...")
        model = download_model(model_id, tmpdir)

        mps_path = os.path.join(tmpdir, f"{model_id}.mps")
        print(f"\n[2] Writing fixed MPS...")
        nRxns, rxn_ids = write_fixed_mps(model, mps_path)

        print(f"\n[3] Running VFWarmup ({2*nRxns} warmup points)...")
        fva_csv = run_vfwarmup(binary, mps_path, nRxns)

        print(f"\n[4] Running COBRApy FVA...")
        cobra_fva = run_cobra_fva(model)

        print(f"\n[5] Comparing results (tol={tol})...")
        ok = compare_fva(fva_csv, cobra_fva, rxn_ids, tol=tol)

    sys.exit(0 if ok else 1)
