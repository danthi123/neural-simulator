import json,glob,os,numpy as np
from ode import *
# fixed points of the undriven apical
vs=np.linspace(-90,10,200001); fn=f_net(vs)
sc=np.flatnonzero(np.sign(fn[:-1])!=np.sign(fn[1:]))
print('fixed points (undriven, g=0):',[round(vs[i],3) for i in sc])
sep=vs[sc[1]]; plat=vs[sc[2]]
m=(vs>sep)&(vs<plat)
print('max net restoring in (sep,plat): %.3f mV-units at v=%.2f'%(fn[m].max(), vs[m][fn[m].argmax()]))
gc=(fn[m]/(vs[m]-EI)).max(); print('static g_ap that removes the plateau (saddle-node): %.4f leak units'%gc)
l,d,r,k=terms(plat,0); print('at plateau v=%.2f: leak %.2f regen %.2f kir %.2f; regen conductance %.3f g_L'%(plat,l,r,k, SR/(1+np.exp(-VHK*(plat-VH)))))
print('GABA current at plateau per unit g_ap: %.2f'%(EI-plat))
# driven fixed point
fd=f_net(vs)+120; i=np.flatnonzero(np.sign(fd[:-1])!=np.sign(fd[1:])); print('driven (D=120,g=0) fp',[round(vs[j],2) for j in i])
