from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts")); sys.path.insert(0, str(HERE))
import table4_tuning as T
import table9_ablation as A
import test_table4_tuning as TT


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root4 = Path(tmp)
        T.write_protocol(root4)
        for t in T.tune_tasks():
            q = TT.quality(t["config_id"], t["seed"], "val")
            TT.fake_run(root4, t, {} if q is None else {pt: q for pt in T.POINTS})
        T.select(root4)                                    
        b = A.base(root4, "stationary_gaussian")
        cs = A.configs(root4, "stationary_gaussian")
        assert len(cs) == 7 and len(A.tasks(root4)) == 35
        shared = [c for c in cs if c["gamma"] == b["gamma"] and c["sigma"] == A.SIGMA0]
        assert len(shared) == 1 and shared[0]["blocks"] == ["gamma", "sigma"]
        assert sorted(c["gamma"] for c in cs if "gamma" in c["blocks"]) == A.GAMMAS
        assert sorted(c["sigma"] for c in cs if "sigma" in c["blocks"]) == A.SIGMAS
        assert all(c["delta0"] == b["delta0"] for c in cs)
        t = next(x for x in A.tasks(root4) if x["config"]["sigma"] == 0.25 and x["seed"] == 3)
        a = A.driver_args(t, "/p.npz")
        assert a.count("--sigma") == 1 and a[a.index("--sigma") + 1] == "0.25"
        assert a[a.index("--gamma") + 1] == f"{b['gamma']:g}" and a[a.index("--model_seed") + 1] == "3003"
        assert a[a.index("--evaluation_split") + 1] == "test" and a[a.index("--schedulers") + 1] == "MAS-SNR"
    print("All table9_ablation checks passed.")


if __name__ == "__main__":
    main()
