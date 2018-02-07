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

t1=100
t=range(len(xyaw))
plt.plot(t[0:t1],xyaw[0:t1],'r-',t[0:t1],xpitch[0:t1],'g-',t[0:t1],xroll[0:t1],'b-',t[0:t1],y1yaw[0:t1],'r--',t[0:t1],y1pitch[0:t1],'g--',t[0:t1],y1roll[0:t1],'b--')
plt.plot(t[0:t1],xyaw[0:t1],'r-',t[0:t1],xpitch[0:t1],'g-',t[0:t1],xroll[0:t1],'b-',t[0:t1],y2yaw[0:t1],'r--',t[0:t1],y2pitch[0:t1],'g--',t[0:t1],y2roll[0:t1],'b--')
plt.show()

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

