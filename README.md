# HeartSteps V1 — causal excursion effects of activity suggestions

Reanalysis of the HeartSteps V1 micro-randomized trial (MRT), estimating the
proximal causal effect of sending a contextually tailored activity suggestion on
the subsequent 30-minute step count.

The estimator is **weighted and centered least squares (WCLS)** — Boruvka,
Almirall, Witkiewitz & Murphy (2018) — written out in explicit matrix algebra in
[`wcls_primary.py`](wcls_primary.py) rather than delegated to a GEE library, so
every term of the sandwich variance is inspectable.

## Status

| Step | Description | State |
| --- | --- | --- |
| 1 | Reproduce the published primary effect | **done** — matches Klasnja et al. (2019) |
| 2 | Jawbone vs. Google Fit sensor comparison, differential coverage by arm | not started |
| 3 | Moderation by study day, decision slot, notification burden | partial (study day only) |

### Step 1 result

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
python wcls_primary.py
```

Prints the analysis-sample construction, both models, a sensitivity table over
every preprocessing choice, and the side-by-side comparison with the publication.
The script asserts its own reconstruction of `study.day.nogap` against the
dataset's ground truth and fails loudly if it drifts.

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

Four findings that cost real debugging time, recorded so they don't have to be
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
