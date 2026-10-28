import json,glob,os,numpy as np
R='raw/research/findings/raw/_cs_race/f1i/step0/'
def recs(prefix='cal_'):
    out=[]
    for f in sorted(glob.glob(R+prefix+'*.json')):
        if f.endswith('prov.json'): continue
        d=json.load(open(f)); ps=d['per_seed'][0]
        name=os.path.basename(f)[len(prefix):-5]
        cell,seed=name.rsplit('_s',1)
        c1=ps['c1']; s=c1['som']; w=c1['v_apical']['pulsed_words'][0]; pw=s['per_word_ms'][w]
        r=dict(name=name,cell=cell,seed=int(seed),c6=ps['c6']['status'],
               c1=dict(word=w,v=np.array(pw['v_apical_min']),g=np.array(pw['g_ap_max']),som=np.array(s['som_spikes_per_ms']),
                       ev=np.array(s['som_cell_events']).reshape(-1,2)),pairs=[])
        for c3 in ps['c3']:
            s3=c3['som']; X,Y=c3['v_apical']['pulsed_words']
            r['pairs'].append(dict(X=X,Y=Y,vX=np.array(s3['per_word_ms'][X]['v_apical_min']),vXmax=np.array(s3['per_word_ms'][X]['v_apical_max']),
                gX=np.array(s3['per_word_ms'][X]['g_ap_max']),vY=np.array(s3['per_word_ms'][Y]['v_apical_min']),gY=np.array(s3['per_word_ms'][Y]['g_ap_max']),
                som=np.array(s3['som_spikes_per_ms'])))
        out.append(r)
    return out
