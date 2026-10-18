"""
HeartSteps V1 -- Revised step 1: what context is actually available as a
moderator of receptivity, and does a baseline self-efficacy measure exist?

"Receptivity" is operationalised as the causal excursion effect of sending a
suggestion on the proximal (30-min) step count, so a legitimate moderator must
be measured AT OR BEFORE the decision point. A variable recorded after the
notification is a post-treatment variable: conditioning the effect on it breaks
the randomisation and the WCLS estimand no longer has a causal interpretation.

This script classifies every column of suggestions.csv on that criterion and
reports the evidence for each classification, so nothing is taken on trust.

Run after wcls_primary.py; it reuses that module's verified analysis frame.
"""

import numpy as np
import pandas as pd

import wcls_primary as W

# ---------------------------------------------------------------------------
# CLASSIFICATION CONSTANTS
#
# The decisive empirical test for post-treatment status: a variable that is
# NEVER observed when send == 0 cannot have existed before the send decision.
# POST_TREATMENT_OBSERVED_THRESHOLD is the observation rate in the not-sent arm
# below which a variable is treated as post-treatment.
# ---------------------------------------------------------------------------

POST_TREATMENT_OBSERVED_THRESHOLD = 0.50

# A variable with this few distinct values among available points carries no
# information and cannot moderate anything.
MIN_DISTINCT_VALUES = 2

# A low-cardinality variable concentrated above this share at its modal value is
# near-degenerate: nominally variable, but with too few cases away from the mode
# to support an interaction with treatment across 37 participants.
MAX_MODAL_SHARE = 0.95

# Out-of-range sentinel values that the weather feed writes instead of NaN.
# Using these columns raw silently corrupts any analysis: the -1024 sentinel
# alone inflates the SD of dec.temperature from 10.4 to 68.9.
TEMPERATURE_SENTINEL = -1024.0     # 26 rows; a negative kelvin-scale artefact
SNOW_SENTINEL = -1.0               # 29 rows; negative snow depth is impossible
TEMPERATURE_PLAUSIBLE_RANGE = (-50.0, 60.0)   # degrees Celsius

# Categorical context with more levels than this is unusable raw and needs
# collapsing before it can enter a model.
MAX_USABLE_CATEGORY_LEVELS = 12

# Sentinel string used by the weather fields instead of NaN. These columns are
# object dtype purely because of it.
WEATHER_MISSING_SENTINEL = "unknown"

# Columns that are the treatment, the outcome, or an identifier -- excluded from
# moderator consideration by role rather than by timing.
ROLE_EXCLUDED = {
    "user.index", "decision.index", "decision.index.nogap",     # identifiers
    "send", "send.active", "send.sedentary", "is.randomized",   # treatment
    "avail",                                                     # conditioning set
    "sugg.select.utime", "sugg.decision.utime", "sugg.context.utime",
    "sugg.select.update", "sugg.tz", "sugg.gmtoff", "sugg.device.utime",
    "study.date", "intake.date", "travel.start", "travel.end", "travel",
    # columns this project derives in wcls_primary.build_analysis_frame --
    # jbsteps30.log IS the outcome, so it must never be offered as a moderator
    "jbsteps30.log", "jbsteps30pre.log", "jbsteps30.analysis",
}

# Outcome windows (post-decision) and their zero-imputed twins: these ARE the
# outcome, at various windows. Never moderators.
OUTCOME_PREFIXES = ("jbsteps10", "jbsteps30", "jbsteps40", "jbsteps60",
                    "jbsteps90", "jbsteps120", "jbmins10", "jbmins30",
                    "jbmins40", "jbmins60", "jbmins90", "jbmins120",
                    "gfsteps10", "gfsteps30", "gfsteps60",
                    "gfmins10", "gfmins30", "gfmins60")


def is_outcome(col):
    """True for post-decision step aggregates (but not the '*pre' windows)."""
    if "pre" in col:
        return False
    return any(col == p or col == p + ".zero" for p in OUTCOME_PREFIXES)


def classify(av):
    """Classify every column of the available-points frame."""
    rows = []
    for c in av.columns:
        col = av[c]
        n_uniq = col.nunique(dropna=True)
        obs_notsent = col[av["send"] == 0].notna().mean()
        obs_sent = col[av["send"] == 1].notna().mean()
        obs_all = col.notna().mean()

        # numeric parse rate, after neutralising the weather sentinel
        as_num = pd.to_numeric(col.astype(str).replace(WEATHER_MISSING_SENTINEL, np.nan),
                               errors="coerce")
        numeric_ok = as_num.notna().sum() > 0 and not pd.api.types.is_bool_dtype(col)

        if c in ROLE_EXCLUDED:
            verdict, why = "excluded: role", "identifier / treatment / raw timestamp"
        elif is_outcome(c):
            verdict, why = "excluded: outcome", "post-decision step aggregate"
        elif obs_notsent < POST_TREATMENT_OBSERVED_THRESHOLD <= obs_sent:
            verdict = "excluded: POST-treatment"
            why = (f"observed {obs_notsent:.0%} when not sent vs {obs_sent:.0%} when "
                   f"sent -- cannot pre-date the send decision")
        elif n_uniq < MIN_DISTINCT_VALUES:
            verdict = "excluded: degenerate"
            why = f"only {n_uniq} distinct value among available points"
        elif n_uniq <= MAX_USABLE_CATEGORY_LEVELS and (
                col.value_counts(normalize=True).max() > MAX_MODAL_SHARE):
            share = col.value_counts(normalize=True).max()
            mode = col.value_counts().idxmax()
            verdict = "excluded: near-degenerate"
            why = (f"{share:.1%} of available points sit at the modal value "
                   f"{mode!r} (> {MAX_MODAL_SHARE:.0%})")
        elif col.dtype == object and n_uniq > MAX_USABLE_CATEGORY_LEVELS and not numeric_ok:
            verdict = "usable but NEEDS COLLAPSING"
            why = f"{n_uniq} raw levels"
        else:
            verdict = "USABLE moderator"
            why = ""

        rows.append({
            "column": c, "dtype": str(col.dtype), "obs%": round(100 * obs_all, 1),
            "obs%_notsent": round(100 * obs_notsent, 1),
            "obs%_sent": round(100 * obs_sent, 1),
            "n_uniq": n_uniq, "verdict": verdict, "evidence": why,
        })
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 70)

    sugg, users, jb = W.load_raw()
    sugg = W.add_study_day(sugg, users, jb)
    primary = W.build_analysis_frame(sugg)
    av = primary[primary["avail"] == 1]

    print("=" * 100)
    print("CONTEXT VARIABLE INVENTORY -- suggestions.csv")
    print(f"Scope: the {len(av)} available (randomised) decision points, "
          f"{av['user.index'].nunique()} participants")
    print("=" * 100)

    d = classify(av)
    for v in ["USABLE moderator", "usable but NEEDS COLLAPSING",
              "excluded: POST-treatment", "excluded: near-degenerate",
              "excluded: degenerate", "excluded: outcome", "excluded: role"]:
        sub = d[d["verdict"] == v]
        if not len(sub):
            continue
        print(f"\n--- {v}  ({len(sub)} columns) ---")
        cols = ["column", "dtype", "obs%", "obs%_notsent", "obs%_sent", "n_uniq"]
        if v.startswith("excluded: POST") or "COLLAPSING" in v or "degenerate" in v:
            cols.append("evidence")
        if v == "excluded: outcome" or v == "excluded: role":
            print(f"  {len(sub)} columns: " + ", ".join(sub['column']))
        else:
            print(sub[cols].to_string(index=False))

    # ---------------- baseline self-efficacy ----------------
    print("\n" + "=" * 100)
    print("BASELINE SELF-EFFICACY -- users.csv")
    print("=" * 100)
    items = ["selfeff.tired.intake", "selfeff.badmood.intake", "selfeff.notime.intake",
             "selfeff.vaca.intake", "selfeff.precip.intake"]
    scale = "selfeff.intake"
    print(f"\nComposite: {scale}")
    print(users[[scale]].describe().T.to_string())
    print(f"  complete for {users[scale].notna().sum()} of {len(users)} participants")
    exact = bool((users[items].sum(axis=1) == users[scale]).all())
    print(f"  equals the exact sum of its 5 items: {exact}")
    print(f"\nItems (each 1-5):")
    print(users[items].describe().T[["count", "mean", "std", "min", "max"]].to_string())
    print(f"\nPost-study version 'selfeff.exit' exists (n={users['selfeff.exit'].notna().sum()}) "
          "but is measured AFTER the intervention -- not a baseline moderator.")

    # ---------------- what does not exist ----------------
    print("\n" + "=" * 100)
    print("REQUESTED BUT NOT AVAILABLE AS A COLUMN")
    print("=" * 100)
    act = av["recognized.activity"].value_counts(dropna=False)
    print(f"""
1. "Prior activity" as an activity-recognition state
   Column 'recognized.activity' EXISTS but is effectively constant among
   available points: {act.to_dict()}.
   There is no WALKING / ON_FOOT / IN_VEHICLE level, so it cannot moderate.
   -> Use 'jbsteps30pre' (or 40/60pre) as the prior-activity measure instead.

2. "Recent notification burden"
   No such column. It must be CONSTRUCTED from the lagged treatment history,
   e.g. the number of sends in the preceding k decision points per participant.
   This is a derived variable and must be built from 'send' ordered by
   (user.index, decision.index).

3. A clean 3-level location variable
   'dec.location.category' is raw Google Places category strings with
   {av['dec.location.category'].nunique()} distinct values. Only 'home'
   ({int((av['dec.location.category'] == 'home').sum())}) and 'work'
   ({int((av['dec.location.category'] == 'work').sum())}) are clean; the rest are
   comma-separated POI lists. The home/work/other collapse used in the published
   analyses must be constructed.

4. Time of day as a clock time
   Use 'sugg.select.slot' (the participant's 5 chosen slots, 1-5, complete and
   clean). Do NOT use 'sugg.decision.slot': it has 7 levels including
   {int((av['sugg.decision.slot'] == 0).sum())} zeros and
   {int((av['sugg.decision.slot'] == 6).sum())} sixes from timezone glitches.
""")

    # ---------------- weather sentinel ----------------
    print("=" * 100)
    print("DATA-QUALITY NOTE: weather fields")
    print("=" * 100)
    for c in ["dec.windspeed", "dec.precipitation.chance"]:
        n_sent = int((av[c] == WEATHER_MISSING_SENTINEL).sum())
        print(f"  {c:<28} object dtype solely because {n_sent} rows hold the "
              f"string '{WEATHER_MISSING_SENTINEL}'; coerce to numeric")

    t = av["dec.temperature"]
    lo, hi = TEMPERATURE_PLAUSIBLE_RANGE
    n_sentinel = int((t == TEMPERATURE_SENTINEL).sum())
    n_nan = int(t.isna().sum())
    clean = t[(t > lo) & (t < hi)]
    print(f"\n  dec.temperature carries TWO kinds of missing:")
    print(f"    {n_nan:>4} NaN  (the same rows as windspeed == '{WEATHER_MISSING_SENTINEL}')")
    print(f"    {n_sentinel:>4} rows equal to {TEMPERATURE_SENTINEL} -- an out-of-range "
          f"sentinel, NOT a temperature")
    print(f"    -> {n_nan + n_sentinel} of {len(av)} ({100*(n_nan+n_sentinel)/len(av):.1f}%) unusable")
    print(f"    raw   : mean {t.mean():7.2f}  SD {t.std():7.2f}  min {t.min():9.1f}")
    print(f"    clean : mean {clean.mean():7.2f}  SD {clean.std():7.2f}  min {clean.min():9.1f}"
          f"  max {clean.max():.1f}  (degrees Celsius)")

    sn = av["dec.snow"]
    print(f"\n  dec.snow: {int((sn == SNOW_SENTINEL).sum())} rows equal {SNOW_SENTINEL} "
          f"(impossible negative depth), and "
          f"{sn.value_counts(normalize=True).max():.1%} of the rest are exactly 0 "
          f"-> excluded as near-degenerate above.")


if __name__ == "__main__":
    main()
