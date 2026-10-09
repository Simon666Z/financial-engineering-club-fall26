#!/usr/bin/env python3
"""Freeze 48 causal numeric inputs for a bounded TabPFN API comparison.

Run with the existing .venv after the alpha experiment is complete. On this
Mac, set DYLD_LIBRARY_PATH as shown in docs/tabpfn.md before importing models.
The API runner uses its isolated .venv-tabpfn; no core model is retrained here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.models.showcase import make_features, _daily_scores
summary=json.loads((root/'reports/alpha/summary.json').read_text())
features=summary['training']['input_features']
data=pd.read_parquet(root/'data/processed/alpha_model_data.parquet').sort_values(['Date','Ticker']).reset_index(drop=True)
train=data[data.Split.eq('train')].reset_index(drop=True)
x_context=make_features(train,features=features)
known=train.target.notna()
train_known=train.loc[known].reset_index(drop=True)
x_known=x_context.loc[known].reset_index(drop=True)
y_known=_daily_scores(train_known.target,train_known.Date).astype(np.float32)
assert train_known.LabelEndDate.notna().all() and train_known.LabelEndDate.le(pd.Timestamp('2022-12-31')).all()
rng=np.random.default_rng(42)
groups=list(train_known.groupby('Date',sort=True).indices.values())
count=20000
base,remainder=divmod(count,len(groups))
extra=set(rng.permutation(len(groups))[:remainder].tolist())
chosen=np.sort(np.concatenate([rng.choice(indices,size=base+(i in extra),replace=False) for i,indices in enumerate(groups)]))
assert len(chosen)==count
out=root/'data/processed/tabpfn';out.mkdir(parents=True,exist_ok=True)
x_known.iloc[chosen].reset_index(drop=True).to_parquet(out/'train_features.parquet',index=False)
np.save(out/'train_labels.npy',y_known[chosen],allow_pickle=False)
train_known.iloc[chosen][['Date','Ticker']].to_parquet(out/'train_keys.parquet',index=False)
for split in ['validation','test']:
 frame=data[data.Split.eq(split)].reset_index(drop=True)
 make_features(frame,features=features).to_parquet(out/f'{split}_features.parquet',index=False)
 frame[['Date','Ticker']].to_parquet(out/f'{split}_keys.parquet',index=False)
 frame[['Date','Ticker','target','LabelEndDate']].to_parquet(out/f'{split}_targets.parquet',index=False)
manifest={
 'feature_names':x_known.columns.tolist(),'train_end':str(train_known.iloc[chosen].Date.max().date()),
 'train_label_end':str(train_known.iloc[chosen].LabelEndDate.max().date()),'transforms_before_sampling':True,
 'seed':42,'training_target_units':'Centered same-date next-day adjusted-close return rank in [-1,1]',
 'train_rows':count,'available_train_labeled_rows':len(train_known),'sampling_policy':'Fixed seed42, approximately equal rowcount pertrainingday; samplewithin eachday onlyafter computingfullobservablecross-sectionalfeatures and supervisedreturnranks.',
 'raw_snapshot_sha256':summary['dataset']['raw_snapshot_sha256'],
 'alpha_data_sha256':summary['dataset']['sha256'],
 'prepared_alpha_sha256':summary['prepared_sha256'],
 'feature_source_sha256':hashlib.sha256((root/'src/models/showcase.py').read_bytes()).hexdigest(),
 'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared TabPFN context:',len(chosen),'trainingrows and',len(x_known.columns),'inputs. Earlierdatecrosssectionscomputedbeforesampling.')
print('Validation/test rows:',len(data[data.Split.eq('validation')]),len(data[data.Split.eq('test')]))
print('Pilot rows:',pd.read_parquet(out/'validation_keys.parquet').groupby('Date').size().iloc[0])
