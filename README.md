# HeartSteps V1 — predicting receptivity to activity suggestions

Reanalysis of the HeartSteps V1 micro-randomized trial (MRT).

**Main question:** can receptivity to activity suggestions be predicted from
context, and does baseline self-efficacy moderate it? *Receptivity* is the causal
excursion effect of sending a suggestion on the Jawbone step count in the 30
minutes that follow.

The estimator is **weighted and centered least squares (WCLS)** — Boruvka,
Almirall, Witkiewitz & Murphy (2018) — written out in explicit matrix algebra in
[`wcls_primary.py`](wcls_primary.py) rather than delegated to a GEE library, so
every term of the sandwich variance is inspectable.

## Status

| Step | Description | State |
| --- | --- | --- |
| 0 | Reproduce the published primary effect (estimator check) | **done** — matches Klasnja et al. (2019) |
| 1 | Inventory pre-decision context and baseline self-efficacy | **done** — [`context_inventory.py`](context_inventory.py) |
| 2 | Moderation by each context variable, one at a time | not started |
| 3 | Moderation by baseline self-efficacy (between-person, n = 37, exploratory) | not started |
| 4 | Leave-one-participant-out prediction — only if step 2 shows real heterogeneity | not started |
| — | Google Fit as a sensitivity check on the final results | not started |

Jawbone is the primary step measure throughout; Google Fit is a sensitivity check
only.

### Step 0 result: the estimator reproduces the paper

`wcls_primary.py` reproduces every effect size the published abstract reports:

| Quantity | Published | This repo |
| --- | --- | --- |
| Marginal effect on 30-min step count | +14%, *p* = .06 | **+14.0%, *p* = .060** |
| …in steps | ~35 over a 253-step average | **+35.6 over 254.1** |
| Initial (day 0) effect | +66%, 167 steps, *p* < .01 | **+66.1%, +168.0, *p* = .002** |
| Participants analysed | 37 | 37 |

β̂ = 0.1312 (SE 0.0674) on log(steps + 0.5), over 6,061 available decision points.

The effect decays with time on study: +66% on day 0, +12.7% by day 21, and
negative thereafter (interaction *p* = .005).

The same estimator and inference are used for every moderation model below.

### Step 1 result: what can moderate receptivity

A moderator must be measured at or before the decision point. `context_inventory.py`
classifies every column of `suggestions.csv` on that rule using an empirical test:
a column that is never observed when no suggestion was sent cannot pre-date the
send decision. Counts are over the 6,061 available decision points.

**Baseline self-efficacy exists:** `selfeff.intake` (users.csv), complete for all
37 participants, the exact sum of five 1–5 items (`selfeff.tired.intake`,
`selfeff.badmood.intake`, `selfeff.notime.intake`, `selfeff.vaca.intake`,
`selfeff.precip.intake`). Range 8–21, mean 14.5, SD 3.31. `selfeff.exit` (n = 35)
is post-intervention and is not a baseline moderator.

**Usable context moderators:**

| Category | Column | Observed | Note |
| --- | --- | --- | --- |
| Decision slot | `sugg.select.slot` | 100% | 1–5, clean. `sugg.decision.slot` has 16 zeros and 3 sixes from timezone glitches |
| Prior activity | `jbsteps30pre` | 91.1% | also 40- and 60-minute pre-windows |
| Weather | `dec.temperature` | 98.9% | 26 rows hold a `-1024` sentinel (see notes) |
| | `dec.weather.condition` | 99.6% | 19 levels; needs collapsing |
| | `dec.windspeed`, `dec.precipitation.chance` | 100% | strings because of an `'unknown'` sentinel; coerce to numeric |
| Location | `dec.location.category` | 100% | 168 raw Google Places strings; only `home` (1,676) and `work` (979) are clean |
| Habituation | `study.day.nogap` | 100% | 0–41 |
| Other | `sugg.device.since`, `is.prefetch` | 73.3%, 100% | hours since phone use; prefetch flag |

**Requested but not available as a column:**

- *Activity-recognition state.* `recognized.activity` is 95.7% `STILL` and 4.3%
  `UNKNOWN`, with no walking or vehicle level, so it cannot moderate. Prior
  activity is measured by `jbsteps30pre` instead.
- *Recent notification burden.* No column; must be built from lagged `send`.
- *A 3-level location.* The home / work / other collapse must be built from
  `dec.location.category`.

**Excluded:** the whole `response.*` block (16 columns, including
`response.temperature` and `response.location.category`) is observed 0% of the
time when no suggestion was sent, so it is post-treatment. The `tag.*` columns
describe the suggestion content and are likewise post-treatment. `connect`,
`snooze.status` and `intransit` are constant among available points because
`avail` is defined from them. `dec.snow` is 98.3% zeros.

## Data

**The dataset is not in this repository.** It is a separate public release and is
fetched independently. From the repository root:

```bash
git clone https://github.com/klasnja/HeartStepsV1.git
cd HeartStepsV1 && git checkout 3016391de426 && cd ..
```

This must land at `HeartStepsV1/data_files/` relative to the repo root, which is
where `DATA_DIR` in `wcls_primary.py` expects it.

### Verifying you have the same bytes

The analysis was run against commit `3016391de426` (18 Jul 2023, the latest
upstream commit). Confirm the four data files match with `git hash-object`:

| File | Git blob SHA-1 |
| --- | --- |
| `data_files/suggestions.csv` | `67a55efd4d79e1f5646bdb26277f37ec988e2403` |
| `data_files/jbsteps.csv` | `1a8da5a8e3cbd9126cc624f0ff3b59a1570dd4cf` |
| `data_files/gfsteps.csv` | `3fbe7a1e4555282414e579f2ba4f3380856a2b3c` |
| `data_files/users.csv` | `5ac3076f0cbf9c840308e766ea87db80671dada8` |

### Attribution

The HeartSteps V1 data are released by Predrag Klasnja, Susan Murphy, Eric
Hekler, Ambuj Tewari, Lisa Jackson, Shawna Smith, Andy Lee and Nick Seewald
(latest version by Eura Shin and Sorawit Saengkyongam) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), at
<https://github.com/klasnja/HeartStepsV1>. Required citation:

> Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B.,
> Hekler, E. B., & Murphy, S. A. (2019). Efficacy of Contextually Tailored
> Suggestions for Physical Activity: A Micro-randomized Optimization Trial of
> HeartSteps. *Annals of Behavioral Medicine*, 53(6), 573–582.
> https://doi.org/10.1093/abm/kay067

## Running it

Python 3.9.6, with `pandas` 2.3.3, `numpy` 2.0.2, `scipy` 1.13.1.

```bash
python wcls_primary.py       # estimator + reproduction of the published effect
python context_inventory.py  # moderator inventory and data-quality checks
python hs_dataload.py        # descriptive Jawbone vs Google Fit comparison
```

`wcls_primary.py` prints the analysis-sample construction, both models, a
sensitivity table over every preprocessing choice, and the side-by-side
comparison with the publication. It asserts its own reconstruction of
`study.day.nogap` against the dataset's ground truth and fails loudly if it
drifts. The other two scripts reuse its analysis frame.

`hs_dataload.py` is descriptive only: its sent-minus-not-sent differences are raw
means, not causal effects.

## Specification provenance

The model, sample filters, outcome transformation, variance correction and
reference test were read out of the authors' own analysis code at
[StatisticalReinforcementLearningLab/HeartstepsV1Code](https://github.com/StatisticalReinforcementLearningLab/HeartstepsV1Code),
not inferred. Each is quoted at its point of use in `wcls_primary.py`.

- `Main Effects Analysis.R` — `model1`, `model2`, the `analysis.data()` filters
- `workspace.analysis.R` — `study.day.nogap`, `jbsteps30.zero`, `*.log`
- `xgeepack.R` — Mancl–DeRouen (2001) variance correction, Hotelling test

Model 1 is `jbsteps30.log ~ jbsteps30pre.log + I(send - .6)`, clustered on
participant, weighted by availability.

**The randomization probability is a known design constant of 0.6.** Setting the
WCLS numerator p̃ equal to it makes every weight exactly 1.0, so the estimator
reduces to *centered* least squares — the "weighted" half of the method is inert
in this trial, and centering is what removes the confounding. The script asserts
the weights are 1.0 rather than assuming it.

Inference uses the Mancl–DeRouen small-sample correction to the cluster-robust
sandwich, with a Hotelling *F* reference distribution on df₂ = 37 − 3 = 34
degrees of freedom, following Liao et al. (2016). With only 37 clusters this
matters: dropping the correction moves *p* from .060 to .052.

## Notes on the data

Findings that cost real debugging time, recorded so they don't have to be
rediscovered:

1. **Neither step file contains a single zero.** The minimum is 1 step, and there
   are no zero rows in 237,865 Jawbone or 197,524 Google Fit records — both
   sensors log only nonzero minutes. An unworn device and a motionless
   participant are therefore indistinguishable in the raw minute-level data by
   construction, not by accident.
2. **`decision.index.following.dec` is not a 30-minute window.** It tags every
   minute until the *next* decision point: median 26 minutes, maximum 230. A
   `groupby(...).sum()` over it correlates only 0.19 with the true 30-minute
   outcome. Use the `jbsteps30` / `gfsteps30` columns already present in
   `suggestions.csv`.
3. **`study.day.nogap` is absent from the public `suggestions.csv`** and is *not*
   `decision.index.nogap // 5` (55% agreement). It is reconstructed in
   `add_study_day()` from the local decision date minus intake date, less the
   travel-block length for post-travel days; this validates 7,176/7,176 against
   `study.day.nogap.following.dec` in `jbsteps.csv`.
4. **`send`, `avail` and `is.randomized` are booleans in object dtype.** `== 1`
   silently matches nothing. Among *available* points `is.randomized` is
   collinear with `send`, so filtering on it leaves no comparison group; it is
   also set at unavailable points, because randomization ran everywhere while
   delivery was gated on availability.
5. **Weather fields carry out-of-range sentinels.** `dec.temperature` has 26 rows
   equal to `-1024` on top of 64 NaN; used raw, the sentinel inflates its SD from
   10.4 to 68.9 °C. `dec.snow` has 29 rows equal to `-1`. `dec.windspeed` and
   `dec.precipitation.chance` hold the string `'unknown'` instead of NaN.
6. **`response.*` columns look like context but are post-treatment.** They are
   recorded when the participant responds to a notification, so they exist only
   when one was sent.

### Sensor comparison

On the correct 30-minute window, among the 3,239 decision points recorded by both
sensors, Jawbone averages 250.6 steps and Google Fit 133.4 (ratio 1.88),
correlation 0.600. The raw sent-minus-not-sent difference is +6.5 steps by
Jawbone and +7.1 by Google Fit. An earlier version of `hs_dataload.py` reported
+19.6 versus +33.0 because it summed over the whole inter-decision gap (note 2);
on correct windows that gap essentially disappears. Coverage does not differ by
arm (Jawbone 0.909 vs 0.909; Google Fit 0.581 vs 0.576).

### Missing step data

The published analysis imputes missing Jawbone counts as **zero** before taking
logs (`workspace.analysis.R:577`, comment: *"We impute zeros"*), for both the
outcome and the pre-decision control. That conflates an unworn tracker with a
stationary participant.

`MISSING_STEPS_POLICY` selects the behaviour. `"published"` reproduces the paper
and is the default; `"nan"` drops those decision points instead. The choice is
consequential but does not overturn the finding — honouring missingness gives a
*larger* estimate, +16.1% (*p* = .054, n = 5,321).

## Caveats

- The comparison targets above are from the **abstract** of Klasnja et al.
  (2019); the full text is paywalled and was not consulted, so the paper's own
  standard errors and confidence intervals are unverified here.
- The 253-step reference average the paper quotes is not defined in the abstract.
  Days 1–7 among available points gives 254.1, which is almost certainly it, but
  that identification is inferred.
- `intake.date` is taken from the local date of `intake.survey.utime`; the
  authors derived it from a non-public `timeslot` table. Indirectly confirmed by
  the 7,176/7,176 `study.day.nogap` match.
- 86 travel-day rows carry a forward-filled study day in `jbsteps.csv`. They are
  excluded here per the documented travel rule. This is the most consequential
  single filter: retaining travel days moves the estimate to +12.3%, *p* = .089.

## References

- Klasnja, P., et al. (2019). *Annals of Behavioral Medicine*, 53(6), 573–582.
- Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing
  Time-Varying Causal Effect Moderation in Mobile Health. *JASA*, 113(523),
  1112–1121.
- Mancl, L. A., & DeRouen, T. A. (2001). A covariance estimator for GEE with
  improved small-sample properties. *Biometrics*, 57(1), 126–134.
- Liao, P., Klasnja, P., Tewari, A., & Murphy, S. A. (2016). Sample size
  calculations for micro-randomized trials in mHealth. *Statistics in Medicine*,
  35(12), 1944–1971.
