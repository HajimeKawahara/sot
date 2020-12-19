import rottheory as rt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
fontsize=14
matplotlib.rcParams.update({'font.size':fontsize})

N=360
inc=90/180.0*np.pi
Thetaeq=np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin

Theta=np.linspace(0.0,2*np.pi,N)

Nt=10
zetaarr=np.array([20,45,60,90,180-60,180-45,180-20])*np.pi/180
colors = plt.cm.viridis(np.linspace(0,1,len(zetaarr)))
for i,Thetaeq in enumerate(np.linspace(0,2*np.pi,Nt)):
    for j,zeta in enumerate(zetaarr):
        arrr=rt.modmax(Theta,zeta,inc,Thetaeq)
        if i==0:
            plt.plot(Theta,arrr,lw=1,color=colors[j],label="$\zeta$="+str(np.round(zeta*180/np.pi,0))+" deg")
        else:
            plt.plot(Theta,arrr,lw=1,color=colors[j],label="")


plt.legend()
plt.ylim(-7.5,5)
plt.ylabel("Modulation Factor")
plt.xlabel("Orbital Phase $\\Theta$")
plt.savefig("modfac.pdf",bbox_inches="tight", pad_inches=0.0)
plt.show()
