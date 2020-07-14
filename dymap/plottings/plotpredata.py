import numpy as np
import matplotlib.pyplot as plt
import process_dscovr as prds
import matplotlib
fontsize=16
matplotlib.rcParams.update({'font.size':fontsize})

mode="pca"
W, obst, lc, timeobj=prds.lcdscovr("pca",Nreduce=2,Nfin=2435)

data=np.load("../samples/test531/predata.npz")
pred=data["arr_0"]
frames=data["arr_1"]
Ni,Ns=np.shape(pred)
cen=[]
cen90=[]
cen10=[]

for i in range(0,Ni):
    cen.append(np.percentile(pred[i,:],50))
    cen10.append(np.percentile(pred[i,:],10))
    cen90.append(np.percentile(pred[i,:],90))

cen=np.array(cen)
cen10=np.array(cen10)
cen90=np.array(cen90)

print(Ni,Ns)
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(211)
plt.plot(obst,lc,label="data",color="C0")
plt.ylabel("PC1")

ax=fig.add_subplot(212)
plt.plot(obst,lc,label="data",color="C0",ls="dashed")
plt.plot(obst[frames],cen,"s",label="prediction",color="C1")
for i in range(0,Ni):
    plt.plot([obst[frames[i]],obst[frames[i]]],[cen10[i],cen90[i]],color="C1",lw=5,alpha=0.5)
#plt.xlim(obst[frames][0],obst[frames][-1])

plt.xlim(245,obst[frames][-1])
plt.ylim(np.min(cen10)-0.01,np.max(cen90)+0.03)
plt.legend()
plt.xlabel("Time [d]")
plt.ylabel("PC1")
plt.savefig("predata.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
