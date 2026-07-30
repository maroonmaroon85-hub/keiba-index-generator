"""
馬場が推奨に反映されているかの検証と、高カーディナリティ特徴の影響。

(62)までの流れで「モデルは馬場を使っているのか」が未確認だったので測った。
判明したこと: **cond の重要度は0.00%、馬場を書き換えても推奨は0.0%しか変わらない**
＝モデルは馬場状態を完全に無視している。

仮説「母父・調教師・騎手・父（分岐の87%）が馬場のような低カーディナリティ特徴を押し出している」を
検証するため、それらを外した構成やカテゴリ正則化を強めた構成と比較する。

実行: python3 ml/highcard_test.py
"""
import sys, itertools, warnings; sys.path.insert(0,'ml'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import features as F
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of
d=F.to_model(F.load_files()); f=F.build_features(d)
keep=(f['n_prior']>=1)&d['odds'].notna()&(d['odds']>0)
d,f=d[keep].reset_index(drop=True),f[keep].reset_index(drop=True)
top3=(d['finish']<=3).astype(int).to_numpy(); odds=d['odds'].to_numpy(float)
inv=1.0/odds; mkt=inv/pd.Series(inv).groupby(d['raceid']).transform('sum').to_numpy()
fx0,maps=F.encode_categoricals(f); fx0['log_odds']=np.log(odds); fx0['mkt_prob']=mkt
cut=d['date'].quantile(0.3); tr,te=(d['date']<cut).to_numpy(),(d['date']>=cut).to_numpy()
pa=load_payout_a('data/payout/a.csv'); wu=load_wu('data/payout/a.csv')
sub0=d.loc[te,['raceid','umaban']].copy(); sub0['y']=d.loc[te,'date'].dt.year.to_numpy()
sub0['fs']=f.loc[te,'fieldsize'].to_numpy()
VAR={
 '1.現行(全部入り)': ([], PARAMS),
 '2.騎手・調教師を除く': (['jockey','trainer'], PARAMS),
 '3.高カーディナリティ4つ全除去': (['jockey','trainer','sire','damsire'], PARAMS),
 '4.現行+カテゴリ正則化強化': ([], dict(PARAMS, cat_smooth=200, min_data_per_group=500, cat_l2=50)),
}
rng=np.random.default_rng(0)
print(f"{'構成':<26}{'AUC':>8}{'cond重要度':>11}{'馬場書換で変化':>14}{'枠連ROI':>10}{'95%CI':>13}{'三連複BOX4':>11}")
for name,(drop,par) in VAR.items():
    cols=[c for c in fx0.columns if c not in drop]
    cats=[c for c in F.CAT_COLS if c in cols]
    fx=fx0[cols]
    ps=[];aucs=[];imps=[]
    for s in range(3):
        m=lgb.LGBMClassifier(random_state=s,**par); m.fit(fx[tr],top3[tr],categorical_feature=cats)
        p=m.predict_proba(fx[te])[:,1]; ps.append(p); aucs.append(roc_auc_score(top3[te],p))
        imp=pd.Series(m.feature_importances_,index=fx.columns); imps.append(imp/imp.sum()*100)
    cond_imp=np.mean([i.get('cond',0) for i in imps])
    p3=np.mean(ps,axis=0)
    # 馬場を書き換えたときの軸変化率
    mm=lgb.LGBMClassifier(random_state=0,**par); mm.fit(fx[tr],top3[tr],categorical_feature=cats)
    fxte=fx.loc[te].copy(); b=mm.predict_proba(fxte)[:,1]
    chg=[]
    for lab,code in maps['cond'].items():
        g=fxte.copy(); g['cond']=code
        chg.append(mm.predict_proba(g)[:,1])
    s2=sub0.copy(); s2['b']=b
    for i,lab in enumerate(maps['cond']): s2[f'c{i}']=chg[i]
    nch=0;ntot=0
    for rid,g in s2.groupby('raceid',sort=False):
        ntot+=1; bt=g.sort_values('b',ascending=False)['umaban'].iloc[0]
        if any(g.sort_values(f'c{i}',ascending=False)['umaban'].iloc[0]!=bt for i in range(len(maps['cond']))): nch+=1
    s2=sub0.copy(); s2['p']=p3
    rw=[];rs=[]
    for rid,g in s2.groupby('raceid',sort=False):
        gg=g.sort_values('p',ascending=False,kind='mergesort'); nums=gg['umaban'].astype(int).tolist()
        w=wu.get(rid)
        if w and w['wakuren'] and len(nums)>=3:
            fs=int(gg['fs'].iloc[0]); wa=waku_of(nums[0],fs)
            cs=sorted({tuple(sorted((wa,waku_of(h,fs)))) for h in nums[1:3]})
            if cs: rw.append(sum(w['wakuren'].get(c,0) for c in cs)/(len(cs)*100))
        p_=pa.get(rid)
        if p_ and p_['sanrenpuku'] and len(nums)>=9:
            cs=[tuple(sorted(c)) for c in itertools.combinations(nums[:4],3)]
            rs.append(sum(p_['sanrenpuku'].get(c,0) for c in cs)/400)
    rw=np.array(rw); rs=np.array(rs); lo,hi=boot(rw,rng,2000)
    print(f"{name:<26}{np.mean(aucs):>8.4f}{cond_imp:>10.2f}%{nch/ntot*100:>13.1f}%"
          f"{rw.mean()*100:>9.1f}%{f'[{lo:.0f},{hi:.0f}]':>13}{rs.mean()*100:>10.1f}%")
