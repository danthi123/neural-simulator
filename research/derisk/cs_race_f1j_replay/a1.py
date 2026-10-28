import json,glob,os,numpy as np
R='raw/research/findings/raw/_cs_race/f1i/step0/'
VH=-35.0
rows=[]
for f in sorted(glob.glob(R+'cal_*.json')):
    if f.endswith('prov.json'): continue
    d=json.load(open(f)); ps=d['per_seed'][0]
    c1=ps['c1']; s=c1['som']; w=c1['v_apical']['pulsed_words'][0]
    pw=s['per_word_ms'][w]
    vmin=np.array(pw['v_apical_min']); vmax=np.array(pw['v_apical_max']); g=np.array(pw['g_ap_max'])
    som=np.array(s['som_spikes_per_ms'])
    ev=np.array(s['som_cell_events']) if s['som_cell_events'] else np.zeros((0,2),int)
    c6=ps['c6']['status']
    # fall time: first t>=150 where vmax< VH and stays below? use min< VH first
    below=np.flatnonzero(vmin[150:]<VH)
    tfall=int(below[0]+150) if below.size else None
    name=os.path.basename(f)[4:-5]
    rows.append(dict(name=name,c6=c6,tfall=tfall,vmin149=vmin[149],vmax149=vmax[149],spread=float(np.nanmax(vmax-vmin)),
        som_pre=int(som[:150].sum()),som_150_200=int(som[150:200].sum()),som_200p=int(som[200:].sum()),
        first_som=(int(np.flatnonzero(som)[0]) if som.any() else None),
        gmax_150_250=float(g[150:250].max()), g149=float(g[149]), gmax_pre=float(g[:150].max()),
        rate=c1['pulsed_rate_hz'], pp=c1['pulse_phase_spikes']))
import collections
print('n',len(rows), collections.Counter(r['c6'] for r in rows))
for r in rows:
    print(r['name'],r['c6'],r['tfall'],'v149 %.1f/%.1f spread %.2f'%(r['vmin149'],r['vmax149'],r['spread']),'som pre/150-200/200+',r['som_pre'],r['som_150_200'],r['som_200p'],'first',r['first_som'],'g149 %.2f gmax150-250 %.2f gpre %.2f'%(r['g149'],r['gmax_150_250'],r['gmax_pre']),'rate',r['rate'],'pp',r['pp'])
json.dump(rows,open('c1rows.json','w'),default=float)
