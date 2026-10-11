"""
HeartSteps V1 -- load, build the analysis frame, and compare the two step sensors.
Data: https://github.com/klasnja/HeartStepsV1  (public, no application required)
"""
import pandas as pd
from pathlib import Path

DATA = Path("HeartStepsV1/data_files")

sugg  = pd.read_csv(DATA / "suggestions.csv", low_memory=False)
jb    = pd.read_csv(DATA / "jbsteps.csv", low_memory=False)   # Jawbone wristband
gf    = pd.read_csv(DATA / "gfsteps.csv", low_memory=False)   # Google Fit (phone)
users = pd.read_csv(DATA / "users.csv", low_memory=False)

for c in ['send', 'avail']:
    sugg[c] = sugg[c].map({True: 1, False: 0})

def proximal(steps_df, label, id_col='user.index',
                   decision_col='decision.index.following.dec',
                   step_col='steps', min_minutes=20):
    out = (steps_df
           .groupby([id_col, decision_col])
           .agg(**{f'steps_{label}':   (step_col, 'sum'),
                   f'minutes_{label}': (step_col, 'count')})
           .reset_index()
           .rename(columns={decision_col: 'decision.index'}))

    valid = out[f'minutes_{label}'] >= min_minutes
    out[f'valid_{label}'] = valid
    out.loc[~valid, f'steps_{label}'] = float('nan')
    return out

df = (sugg[['user.index','decision.index','send','avail','sugg.decision.slot']]
      .merge(proximal(jb, 'jawbone'),   on=['user.index','decision.index'], how='left')
      .merge(proximal(gf, 'googlefit'), on=['user.index','decision.index'], how='left'))

# Randomization occurs only when the participant is available (per README)
a = df[df['avail'] == 1]
print(f"Analysis set: {len(a)} available decision points "
      f"({int((a['send']==1).sum())} sent, {int((a['send']==0).sum())} not sent)")
print(f"Randomization probability: {(a['send']==1).mean():.3f}\n")

for col in ['steps_jawbone', 'steps_googlefit']:
    m = a.dropna(subset=[col])
    s1 = m.loc[m['send']==1, col].mean()
    s0 = m.loc[m['send']==0, col].mean()
    print(f"{col:<17} sent {s1:7.1f} | not sent {s0:7.1f} | difference {s1-s0:+7.1f}   n={len(m)}")

both = a.dropna(subset=['steps_jawbone','steps_googlefit'])
d_jb = (both.loc[both['send']==1,'steps_jawbone'].mean()
        - both.loc[both['send']==0,'steps_jawbone'].mean())
d_gf = (both.loc[both['send']==1,'steps_googlefit'].mean()
        - both.loc[both['send']==0,'steps_googlefit'].mean())

print(f"\nSame {len(both)} decision points measured by both sensors:")
print(f"  correlation              {both['steps_jawbone'].corr(both['steps_googlefit']):.3f}")
print(f"  mean Jawbone             {both['steps_jawbone'].mean():7.1f}")
print(f"  mean Google Fit          {both['steps_googlefit'].mean():7.1f}")
print(f"  effect by Jawbone        {d_jb:+7.1f} steps")
print(f"  effect by Google Fit     {d_gf:+7.1f} steps")