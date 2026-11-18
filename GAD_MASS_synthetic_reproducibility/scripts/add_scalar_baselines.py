import json
import math
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    u = json.load(open(src))
    delta = float(u.get("delta", 1e-3))
    for ds, entry in u["datasets"].items():
        mass = entry.get("MAS-SNR")
        if not mass:
            raise SystemExit(f"{ds}: no MASS configuration in {src}")
        gamma = float(mass["gamma"])
        cbar = math.log(1.0 / delta) / (2.0 * gamma)
        rule = "scalar baseline for Tables III/IV: MASS gamma, delta-complete cbar_T, paper shape"
        if not entry.get("Cosine"):
            entry["Cosine"] = {"config_id": f"cosine_rule_gamma{gamma:g}", "gamma": gamma, "cosine_s": 0.008,
                               "cosine_terminal_epsilon": 1e-3, "cbar_T": cbar, "note": rule}
        if not entry.get("FCPS"):
            entry["FCPS"] = {"gamma": gamma, "cbar_T": cbar, "fcps_alpha": 4.0, "fcps_cmin_table3": 0.1,
                             "fcps_cmin_table4": 1e-4, "note": rule}
        print(f"{ds}: gamma = {gamma:g}, cbar_T = {cbar:.1f} for FCPS and Cosine")
    u["scalar_baseline_rule"] = ("FCPS and Cosine: per dataset the selected MASS gamma, cbar_T = log(1/delta)/(2 gamma), "
                                 "Cosine s=0.008 eps=1e-3, FCPS alpha=4 (not tuned; no Cosine was stable under EM in Table V)")
    json.dump(u, open(dst, "w"), indent=2)
    print(f"written: {dst}")


if __name__ == "__main__":
    main()
