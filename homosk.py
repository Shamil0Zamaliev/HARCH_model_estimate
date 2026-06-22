import numpy as np, pandas as pd, json, math
def _gser(a,x):
    ap=a;s=1.0/a;d=s
    for _ in range(600):
        ap+=1;d*=x/ap;s+=d
        if abs(d)<abs(s)*1e-15:break
    return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
def _gcf(a,x):
    tiny=1e-300;b=x+1-a;c=1/tiny;d=1/b;hh=d
    for i in range(1,600):
        an=-i*(i-a);b+=2;d=an*d+b
        if abs(d)<tiny:d=tiny
        c=b+an/c
        if abs(c)<tiny:c=tiny
        d=1/d;de=d*c;hh*=de
        if abs(de-1)<1e-15:break
    return math.exp(-x+a*math.log(x)-math.lgamma(a))*hh
def chi2_sf(x,k):
    x=float(x);a=k/2.;y=x/2.
    if x<=0:return 1.0
    return 1.-_gser(a,y) if y<a+1 else _gcf(a,y)
OUT="G:\\Documents\\Claude\\Projects\\Harch(2)"
s=json.load(open(OUT+"/figs/ch2_summary.json"))
a0,a1,a2=s['mle']['a0'],s['mle']['a1'],s['mle']['a2']
df=pd.read_csv(OUT+"/USDCB_250128_260228.csv",sep=";");df.columns=[c.strip("<>").upper() for c in df.columns]
close=df["CLOSE"].astype(float).values;h=np.diff(np.log(close));h=h-h.mean();N=len(h)
s2=np.empty(N)
for t in range(N):
    aa=h[t-1] if t>=1 else 0.;bb=h[t-2] if t>=2 else 0.
    s2[t]=a0+a1*aa**2+a2*(aa+bb)**2
eps=h/np.sqrt(s2)
def khal(x,m):
    x=x-x.mean(); s2g=np.mean(x**2); kap=np.mean(x**4)/s2g**2   # выборочный эксцесс (не избыточный)
    sk=x**2/(4*s2g)            # масштаб c=N/(4 sigma^2): сумма приращений = N/4
    C=np.cumsum(sk); s2y=np.mean(sk)   # = 1/4
    tau=np.empty(m); idx=0
    for th in range(1,m+1):
        while idx<N and C[idx]<th: idx+=1
        tau[th-1]=idx+1 if idx<N else N
    thr=np.arange(1,m+1)
    raw=np.sum((s2y/thr)*(tau-thr/s2y)**2)   # исходная статистика Халиуллина
    stat=raw/(kap-1.0)                        # нормировка на эксцесс приращений -> chi2_{m-2}
    return raw,stat,kap,m-2,chi2_sf(stat,m-2)
for m in (round(N/4),round(N/5)):
    print("=== m=%d (df=%d) ==="%(m,m-2))
    for nm,x in (("h   ",h),("eps ",eps)):
        raw,st,kap,d,p=khal(x,m)
        print("  %s kappa=%.2f  raw=%.1f  chi2=%.1f  p=%.4f"%(nm,kap,raw,st,p))
