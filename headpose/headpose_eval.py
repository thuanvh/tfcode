import numpy as np
import matplotlib.pyplot as plt

folder = "D:\\sandbox\\vmakeup\\VirtualMakeover\\Binaries\\Bin\\"
file = folder+"out1.csv"
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
fig = plt.figure(1)
plt.plot(xyaw,y1yaw,'ro')
fig.savefig(folder+"y1yaw.png")
fig = plt.figure(2)
plt.plot(xpitch,y1pitch,'ro')
plt.savefig(folder+"y1pitch.png")
plt.figure(3)
plt.plot(xroll,y1roll,'ro')
plt.savefig(folder+"y1roll.png")
plt.figure(4)
plt.plot(xyaw,y2yaw,'ro')
plt.savefig(folder+"y2yaw.png")
plt.figure(5)
plt.plot(xpitch,y2pitch,'ro')
plt.savefig(folder+"y2pitch.png")
plt.figure(6)
plt.plot(xroll,y2roll,'ro')
plt.savefig(folder+"y2roll.png")
