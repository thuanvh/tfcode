import numpy as np
import matplotlib.pyplot as plt

folder = "D:\\sandbox\\vmakeup\\VirtualMakeover\\Binaries\\Bin\\"
file = folder+"out2.csv"
data = np.genfromtxt(file,delimiter=',')
xyaw=data[:,1]
xpitch=data[:,2]
xroll=data[:,3]
y1yaw=data[:,4]
y1pitch=data[:,5]
y1roll=data[:,6]
y2yaw=data[:,7]
y2pitch=data[:,8]
y2roll=data[:,9]
plt.figure(figsize=(20,10))
plt.subplot(321)
plt.plot(xyaw,y1yaw,'r,',xyaw,xyaw,'b-')
plt.legend(["yaw1","truth"])
#plt.savefig(folder+"y1yaw.png")
plt.subplot(323)
plt.plot(xpitch,y1pitch,'r,',xpitch,xpitch,'b-')
plt.legend(["pitch1","truth"])
#plt.savefig(folder+"y1pitch.png")
plt.subplot(325)
plt.plot(xroll,y1roll,'r,',xroll,xroll,'b-')
plt.legend(["roll1","truth"])
#plt.savefig(folder+"y1roll.png")
plt.subplot(322)
plt.plot(xyaw,y2yaw,'g,',xyaw,xyaw,'b-')
plt.legend(["yaw2","truth"])
#plt.savefig(folder+"y2yaw.png")
plt.subplot(324)
plt.plot(xpitch,y2pitch,'g,',xpitch,xpitch,'b-')
plt.legend(["pitch2","truth"])
#plt.savefig(folder+"y2pitch.png")
plt.subplot(326)
plt.plot(xroll,y2roll,'g,',xroll,xroll,'b-')
plt.legend(["roll2","truth"])
#plt.savefig(folder+"y2roll.png")
plt.savefig(folder+"all.png")

t0=len(xyaw)-1000
t1=len(xyaw)
tr=slice(t0,t1)
t=range(len(xyaw))
plt.plot(t[tr],xyaw[tr],'r-',t[tr],xpitch[tr],'g-',t[tr],xroll[tr],'b-',t[tr],y1yaw[tr],'r--',t[tr],y1pitch[tr],'g--',t[tr],y1roll[tr],'b--')
#plt.plot(t[tr],xyaw[tr],'r-',t[tr],xpitch[tr],'g-',t[tr],xroll[tr],'b-',t[tr],y2yaw[tr],'r--',t[tr],y2pitch[tr],'g--',t[tr],y2roll[tr],'b--')
plt.show()

file="out7.csv"
data = np.genfromtxt(file,delimiter=',')
xyaw=data[:,1]
xpitch=data[:,2]
xroll=data[:,3]
y1yaw=data[:,4]
y1pitch=data[:,5]
y1roll=data[:,6]
plt.figure(figsize=(20,10))
plt.subplot(311)
plt.plot(xyaw,y1yaw,'r,',xyaw,xyaw,'b-')
plt.legend(["yaw1","truth"])
#plt.savefig(folder+"y1yaw.png")
plt.subplot(312)
plt.plot(xpitch,y1pitch,'r,',xpitch,xpitch,'b-')
plt.legend(["pitch1","truth"])
#plt.savefig(folder+"y1pitch.png")
plt.subplot(313)
plt.plot(xroll,y1roll,'r,',xroll,xroll,'b-')
plt.legend(["roll1","truth"])
#plt.savefig(folder+"y1roll.png")
plt.show()

