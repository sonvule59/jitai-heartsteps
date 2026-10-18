"""
HeartSteps V1 -- descriptive comparison of the two step sensors on the proximal
30-minute window.

Jawbone (wristband) is the primary step measure; Google Fit (phone) is used only
as a sensitivity check. This script is DESCRIPTIVE: the sent-minus-not-sent
differences below are raw means, not causal excursion effects. Causal estimates
come from the WCLS estimator in wcls_primary.py.

Earlier versions of this file summed minute-level steps by
`decision.index.following.dec`. That column tags every minute until the NEXT
decision point (median 26 min, max 230), not a 30-minute window, so those
sensor numbers were wrong. The proximal outcome is now taken from the
`jbsteps30` / `gfsteps30` columns shipped in suggestions.csv.

Data: https://github.com/klasnja/HeartStepsV1 @ 3016391 (CC BY 4.0)
"""
import numpy as np

import wcls_primary as W

# Missing step data stay missing: an unworn device is not zero steps.
# "nan" keeps NaN; "published" would zero-impute as Klasnja et al. (2019) did.
MISSING_STEPS_POLICY = "nan"

# Proximal outcome columns for each sensor (30 min after the decision point).
SENSOR_COLUMNS = {"Jawbone": "jbsteps30", "Google Fit": "gfsteps30"}


def build_frame():
    """Available decision points under the published sample filters."""
    sugg, users, jb = W.load_raw()
    sugg = W.add_study_day(sugg, users, jb)
    frame = W.build_analysis_frame(sugg, missing_policy=MISSING_STEPS_POLICY)
    return frame[frame["avail"] == 1].copy()


def main():
    a = build_frame()
    print(f"Analysis set: {len(a)} available decision points "
          f"({int((a['send'] == 1).sum())} sent, {int((a['send'] == 0).sum())} not sent), "
          f"{a['user.index'].nunique()} participants")
    print(f"Realized P(send | available): {a['send'].mean():.4f} "
          f"(design constant {W.RANDOMIZATION_PROB})\n")

    print("Coverage and raw means per sensor (NaN = no device data; not imputed):")
    for label, col in SENSOR_COLUMNS.items():
        m = a.dropna(subset=[col])
        cov1 = a.loc[a["send"] == 1, col].notna().mean()
        cov0 = a.loc[a["send"] == 0, col].notna().mean()
        s1 = m.loc[m["send"] == 1, col].mean()
        s0 = m.loc[m["send"] == 0, col].mean()
        print(f"  {label:<11} coverage sent {cov1:.3f} / not sent {cov0:.3f} | "
              f"mean sent {s1:6.1f} / not sent {s0:6.1f} | raw diff {s1 - s0:+6.1f}  "
              f"n={len(m)}")

    jb, gf = SENSOR_COLUMNS["Jawbone"], SENSOR_COLUMNS["Google Fit"]
    both = a.dropna(subset=[jb, gf])
    raw_diff = {c: both.loc[both["send"] == 1, c].mean()
                   - both.loc[both["send"] == 0, c].mean() for c in (jb, gf)}

    print(f"\nSame {len(both)} decision points recorded by both sensors:")
    print(f"  mean Jawbone             {both[jb].mean():7.1f}")
    print(f"  mean Google Fit          {both[gf].mean():7.1f}")
    print(f"  ratio Jawbone/Google Fit {both[jb].mean() / both[gf].mean():7.2f}")
    print(f"  correlation (raw)        {both[jb].corr(both[gf]):7.3f}")
    print(f"  correlation (log +0.5)   "
          f"{np.log(both[jb] + W.LOG_OFFSET).corr(np.log(both[gf] + W.LOG_OFFSET)):7.3f}")
    print(f"  raw diff by Jawbone      {raw_diff[jb]:+7.1f} steps")
    print(f"  raw diff by Google Fit   {raw_diff[gf]:+7.1f} steps")
    print("\n  (raw differences only -- see wcls_primary.py for causal estimates)")


if __name__ == "__main__":
    main()
