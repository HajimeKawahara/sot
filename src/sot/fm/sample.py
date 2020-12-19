import rottheory as rt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
fontsize=14
matplotlib.rcParams.update({'font.size':fontsize})

N=360
inc=45.0/180.0*np.pi
Thetaeq=np.pi
zeta=23.4/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin

Theta=np.linspace(0.0,2*np.pi,N)
arrp=rt.modmax(Theta,zeta,inc,Thetaeq)
arrr=rt.modmax(Theta,np.pi-zeta,inc,0.0)

Ns=90
t = np.arange(Ns)
colors = plt.cm.viridis(np.linspace(0,1,Ns))

for i,Thetaeq in enumerate(np.linspace(0,2*np.pi,Ns)):
    arrr=rt.modmax(Theta,np.pi-zeta,inc,Thetaeq)
    plt.plot(Theta,arrr,lw=1,color=colors[i])

plt.plot(Theta,arrr,lw=3,color="red",label="$\\zeta=156.6^\\circ, \\Theta_\\mathrm{eq} = 0^\\circ$")
plt.plot(Theta,arrp,lw=3,color="black",label="$\\zeta=23.4^\\circ, \\Theta_\\mathrm{eq} = 180^\\circ$")

plt.legend()
plt.ylim(-7.5,5)
plt.ylabel("Modulation Factor")
plt.xlabel("Orbital Phase $\\Theta$")
plt.savefig("modfac.pdf",bbox_inches="tight", pad_inches=0.0)
plt.show()
