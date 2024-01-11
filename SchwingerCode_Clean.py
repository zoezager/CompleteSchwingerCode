#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy
from matplotlib import pyplot
from scipy.signal import hilbert
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import scipy.special as special 
import scipy.integrate as integrate 

numpy.set_printoptions(precision=3)


# Creating the Mesh

# In[70]:


#x values
J = 2000 #number of points
systemsize = 1000
dx = float(systemsize)/float(J)
x_grid = numpy.zeros(J)
for i in range(J):
    x_grid[i] = (i-J/2)*dx

#t values
T = 500 #final timestep
N = 1000 #number of timesteps
dt = T/N #T/N
t_grid = numpy.array([n*dt for n in range(N)])


# Physical Constants/Parameters

# In[71]:


Q = 4 #charge of fixed points
L = 40 #total capacitor size
g = 2.5 #"effective mass"
r = dt*dt/(2*dx*dx)


# Initial condition for bosonic function $\phi = 0$ and $\partial_t \phi = 0$

# In[72]:


u_initial = numpy.zeros(J)


# The initial condition for the field due to the fixed initial charges is $Q\left[ \Theta \left( x + \frac{L}{2} \right) - \Theta \left( x - \frac{L}{2} \right) \right]$

# In[73]:


pyplot.plot(x_grid, Q*(numpy.heaviside(x_grid + L/2,1) - numpy.heaviside(x_grid - L/2,1)))
pyplot.xlim(-30,30)


# So the initial charge distribution is found by $\rho_0 = -\partial_x E_0$

# In[74]:


initialfield = Q*(numpy.heaviside(x_grid + L/2,1) - numpy.heaviside(x_grid - L/2,1))
p_initial = -numpy.gradient(initialfield, dx)
pyplot.plot(x_grid,p_initial)
pyplot.xlim(-30,30)


# Define matrices to implement system of equations for finite difference

# In[75]:


minusrarray = numpy.array([-r for i in range(J-1)])
plusrarray = numpy.array([r for i in range(J-1)])
firstdiag = numpy.array([1+2*r for i in range(J)])
seconddiag = numpy.array([2-2*r for i in range(J)])
identity = numpy.array([1 for i in range(J)])

nplus_array = numpy.diagflat(minusrarray,-1) + numpy.diagflat(firstdiag) + numpy.diagflat(minusrarray,1)
n_array = numpy.diagflat(plusrarray,-1) + numpy.diagflat(seconddiag) + numpy.diagflat(plusrarray,1)


# Define $f(\phi)$ at each time step, a function that takes into account the parts of the system of equation not represented by the matrices above

# In[76]:


def fvector(U, x_grid):
    result = numpy.zeros(J)
    for j in range(J):
        result[j] = g*g*dt*dt*U[j] + g*Q*dt*dt*(numpy.heaviside(x_grid[j] + L/2,1) - numpy.heaviside(x_grid[j] - L/2,1))
    return result


# Solve the system

# In[77]:


U_record = [] #keeps track of phi for each time step
U_record.append(u_initial.copy())
U_record.append(u_initial.copy())

#implementing initial value condition for phi and derivative of phi
U_minus = u_initial 
U = u_initial 

#solving the system
for i in range(2,N):
    fvec = fvector(U, x_grid)
    U_plus = numpy.linalg.solve(nplus_array, n_array.dot(U) - U_minus - fvec)
    U_minus = U
    U = U_plus
    U_record.append(U)


# Current is defined as $j^1 = g \partial_t \phi$.  We plot this at $x=0$ as a function of time.

# In[78]:


numpy.shape(U_record)


# In[79]:


U_record = numpy.array(U_record)
timedependent = U_record[:,int(J/2)]
jx = g*numpy.gradient(timedependent, dt)

pyplot.plot(t_grid,jx)
pyplot.xlabel('Time')
pyplot.ylabel('Current')
pyplot.title('Current at x=0, Setup I')
pyplot.ylim(-15,15)
pyplot.show()


# Integrating Maxwell's equation $\partial_\mu F^{\mu\nu} = j^\nu$ allows us to find the electric field due to the generated plus the initial charges $F^{10} = g \phi + Q \left[ \Theta \left( x + \frac{L}{2} \right) - \Theta \left( x - \frac{L}{2} \right) \right]$

# In[80]:


phi0 = U_record[:,int(J/2)]
efield = g*phi0 + Q*(numpy.heaviside(L/2,1) - numpy.heaviside(-L/2,1))

pyplot.plot(t_grid, efield)
pyplot.xlabel('Time')
pyplot.ylabel('Electric Field')
pyplot.title('Electric Field at x=0, Setup I')
pyplot.show()


# We can also plot the bosonic field $\phi$

# In[81]:


pyplot.plot(t_grid, phi0)
pyplot.xlabel('Time')
pyplot.ylabel('Phi')
pyplot.title('Phi at x=0')
pyplot.show()


# In[82]:


#Find the Current
current_record = []

for x in range(J): #loop over all x values
    timedependent = U_record[:,x] #take the time dependent part at the x value we are currently looking at
    jx = g*numpy.gradient(timedependent, dt) #take the derivative wrt t 
    current_record.append(jx)

current_record = numpy.array(current_record)


# In[83]:


#Find the Charge
charge_record = []
charge_record_without_initial = []

for t in range(N):
    xdependent = U_record[t]
    px = -g*numpy.gradient(xdependent, dx) + p_initial
    charge_record_without_initial.append(-g*numpy.gradient(xdependent, dx))
    charge_record.append(px)

charge_record = numpy.array(charge_record)
charge_record_without_initial = numpy.array(charge_record_without_initial)


# In[84]:


pyplot.plot(x_grid, current_record[:,int(N/2)], label="current", color='green')
pyplot.plot(x_grid, charge_record[int(N/2)], label="charge", color='red')
pyplot.plot(x_grid, g*U_record[int(N/2)] + Q*(numpy.heaviside(x_grid + L/2,1) - numpy.heaviside(x_grid - L/2,1)), label="field", color='orange')

pyplot.xlabel('x')
pyplot.ylabel('Current,field,charge')
pyplot.title('Current, field and charge over space at t=250')
pyplot.xlim(-100,100)
pyplot.legend()
pyplot.show() #current and field are now exactly the same


# Now, we create movies to see the dynamical evolution of the field, charge, and current.  The current is

# In[35]:


#Current Movie


 
# duration of the video
duration = 30
 
# matplot subplot
fig, ax = pyplot.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x_grid, current_record[:,int(t*10)])
    ax.vlines(-20,-20,20, linestyles='dotted')
    ax.vlines(20,-20,20, linestyles='dotted')
    ax.set_ylim(-20, 20)
    ax.set_xlim(-100,100)
    ax.title.set_text("Current, setup I")
     
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = 100, loop = True, autoplay = True)


# In[36]:


numpy.shape(current_record)


# The charge is given by the generated charges $j_\phi^0$ and the external initial charges $j^0_{ext}$.  The resulting expression is 
# $j^0 = j_\phi^0 + j_{ext}^0 = -g\partial_x \phi + Q\left[ \delta\left( x -\frac{L}{2} \right)-\delta \left( x+\frac{L}{2} \right) \right]$

# In[37]:


#Charge Movie

t_gridnew = t_grid[:-1]
x_gridnew = x_grid[:-1]


 
# duration of the video
duration = 30
 
# matplot subplot
fig, ax = pyplot.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x_grid, charge_record[int(t*10)])
    ax.set_ylim(-4.1,4.1)
    ax.set_xlim(-125,125)
    ax.vlines(-20,-20,20, linestyles='dotted')
    ax.vlines(20,-20,20, linestyles='dotted')
    ax.title.set_text("Charge, setup I")
     
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = 200, loop = True, autoplay = True)


# In[38]:


print(numpy.shape(charge_record))
print(numpy.shape(charge_record_without_initial))


# And the Electric field is given by $E = g\phi + Q\left[ \Theta \left(x+\frac{L}{2}\right)-\Theta \left( x - \frac{L}{2} \right) \right]$

# In[39]:


#E-field movie

# duration of the video
duration = 49
 
# matplot subplot
fig, ax = pyplot.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x_grid, g*U_record[int(t*10)] + Q*(numpy.heaviside(x_grid + L/2,1) - numpy.heaviside(x_grid - L/2,1)))
    ax.title.set_text("E-field, setup I")
    ax.set_ylim(-8, 8)
    ax.set_xlim(-100,100)
    ax.vlines(-20,-20,20, linestyles='dotted')
    ax.vlines(20,-20,20, linestyles='dotted')
     
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = 100, loop = True, autoplay = True)


# We create an array which stores the values of the external field for all time (it is always constant)

# In[85]:


extfield = []
temp = []

for x in range(J): #space
    temp.append(Q*(numpy.heaviside(x_grid[x] + L/2,1) - numpy.heaviside(x_grid[x] - L/2,1)))
temp = numpy.array(temp)

for i in range(N): #time
    extfield.append(temp)
extfield = numpy.array(extfield)
numpy.shape(extfield)


# The energy for the system can be summarized by the following Hamiltonian
# $H = \int dx \left( \frac{1}{2}(\partial_t\phi)^2+\frac{1}{2}(\partial_x\phi)^2+\frac{1}{2}E_x^2 \right)$
# where $E_x = g\phi + Q_{ext}$

# In[86]:


#Hamiltonian

term1 = numpy.square(charge_record_without_initial)/(2*g*g) #charge energy
print(numpy.shape(term1))
term2 = numpy.square(current_record)/(2*g*g) #current energy
print(numpy.shape(term2))
term3 = .5*numpy.square(numpy.add(g*U_record, extfield)) #e-field energy
print(numpy.shape(term3))


# In[87]:


#Integration from 0 to L

integration1 = []
integration2 = []
integration3 = []
density1 = [] #charge
density2 = [] #current
density3 = [] #e-field

for t in range(N-1): #looping over times
    xdepend1 = term1[t]
    density1.append(xdepend1)
    xdepend2 = term2[:,t]
    density2.append(xdepend2)
    xdepend3 = term3[t]
    density3.append(xdepend3)
    term1int = numpy.trapz(xdepend1,x_grid)
    term2int = numpy.trapz(xdepend2,x_grid)
    term3int = numpy.trapz(xdepend3,x_grid)
    integration1.append(term1int)
    integration2.append(term2int)
    integration3.append(term3int)

t_gridnew = t_grid[:-1]

print(numpy.shape(density1))
print(numpy.shape(density2))
print(numpy.shape(density3))
print(numpy.shape(integration1))
print(numpy.shape(integration2))
print(numpy.shape(integration3))


# In[90]:


total = numpy.add(numpy.add(integration2, integration3), integration1)
pyplot.plot(t_gridnew, total, label='total', color='blue')
pyplot.plot(t_gridnew, integration3, label="field", color='orange')
pyplot.plot(t_gridnew, integration2, label="current", color='green')
pyplot.plot(t_gridnew, integration1, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Energies as a function of time over all space')
pyplot.ylim(0,500)
pyplot.xlim(0,500)
pyplot.legend()
pyplot.show() 


# In[89]:


average1 = []
average2 = []
average3 = []
start = 0

for i in range(199):
    ave1 = 0
    ave2 = 0
    ave3 = 0
    for j in range(5):
        ave1 = ave1 + integration1[start + j]
        ave2 = ave2 + integration2[start + j]
        ave3 = ave3 + integration3[start + j]
    ave1 = ave1/5
    ave2 = ave2/5
    ave3 = ave3/5
    start = start + 5
    average1.append(ave1)
    average2.append(ave2)
    average3.append(ave3)

average1 = numpy.array(average1)
average2 = numpy.array(average2)
average3 = numpy.array(average3)

times = numpy.array([2.5*n for n in range(199)])

total = numpy.add(numpy.add(average3, average2), average1)

pyplot.plot(times, total, label="total", color='blue')
pyplot.plot(times, average3, label="field", color='orange')
pyplot.plot(times, average2, label="current", color='green')
pyplot.plot(times, average1, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Time-averaged energies as a function of time over all space')
pyplot.ylim(0,500)
pyplot.xlim(0,500)
pyplot.legend()
pyplot.show() 


# We can separate out the energies inside of and outside of the capacitor

# In[32]:


integration1cap = []
integration2cap = []
integration3cap = []
density1 = [] #charge
density2 = [] #current
density3 = [] #e-field
x_gridslice = x_grid[960:1041]

for i in range(N-1):
    xdepend1 = term1[i]
    density1.append(xdepend1)
    xdepend1 = xdepend1[960:1041]
    xdepend2 = term2[:,i]
    density2.append(xdepend2)
    xdepend2 = xdepend2[960:1041]
    xdepend3 = term3[i]
    density3.append(xdepend3)
    xdepend3 = xdepend3[960:1041]
    term1int = numpy.trapz(xdepend1,x_gridslice)
    term2int = numpy.trapz(xdepend2,x_gridslice)
    term3int = numpy.trapz(xdepend3,x_gridslice)
    integration1cap.append(term1int)
    integration2cap.append(term2int)
    integration3cap.append(term3int)

t_gridnew = t_grid[:-1]


pyplot.plot(t_gridnew, integration3cap, label="field", color='orange')
pyplot.plot(t_gridnew, integration2cap, label="current", color='green')
pyplot.plot(t_gridnew, integration1cap, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Energies as a function of time in the capacitor')
pyplot.xlim(0,4000)
pyplot.legend()
pyplot.show()


# In[33]:


difference1 = numpy.subtract(integration1, integration1cap)
difference2 = numpy.subtract(integration2, integration2cap)
difference3 = numpy.subtract(integration3, integration3cap)

pyplot.plot(t_gridnew, difference3, label="field", color='orange')
pyplot.plot(t_gridnew, difference2, label="current", color='green')
pyplot.plot(t_gridnew, difference1, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Energies as a function of time outside the capacitor')
pyplot.xlim(0,4000)
pyplot.ylim(0,750)
pyplot.legend()
pyplot.show()


# In[34]:


integration1cap = []
integration2cap = []
integration3cap = []
density1 = [] #charge
density2 = [] #current
density3 = [] #e-field
x_gridslice = x_grid[920:1081]

for i in range(N-1):
    xdepend1 = term1[i]
    density1.append(xdepend1)
    xdepend1 = xdepend1[920:1081]
    xdepend2 = term2[:,i]
    density2.append(xdepend2)
    xdepend2 = xdepend2[920:1081]
    xdepend3 = term3[i]
    density3.append(xdepend3)
    xdepend3 = xdepend3[920:1081]
    term1int = numpy.trapz(xdepend1,x_gridslice)
    term2int = numpy.trapz(xdepend2,x_gridslice)
    term3int = numpy.trapz(xdepend3,x_gridslice)
    integration1cap.append(term1int)
    integration2cap.append(term2int)
    integration3cap.append(term3int)

t_gridnew = t_grid[:-1]


pyplot.plot(t_gridnew, integration3cap, label="field", color='orange')
pyplot.plot(t_gridnew, integration2cap, label="current", color='green')
pyplot.plot(t_gridnew, integration1cap, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Energies as a function of time in 2x capacitor')
pyplot.xlim(0,500)
pyplot.legend()
pyplot.show()


# In[35]:


difference1 = numpy.subtract(integration1, integration1cap)
difference2 = numpy.subtract(integration2, integration2cap)
difference3 = numpy.subtract(integration3, integration3cap)

pyplot.plot(t_gridnew, difference3, label="field", color='orange')
pyplot.plot(t_gridnew, difference2, label="current", color='green')
pyplot.plot(t_gridnew, difference1, label="charge", color='red')

pyplot.xlabel('Time')
pyplot.ylabel('Energy')
pyplot.title('Energies as a function of time outside 2x capacitor')
pyplot.xlim(0,4000)
pyplot.ylim(0,750)
pyplot.legend()
pyplot.show()


# PEFORMING CORRECTION CHECKS BELOW HERE

# First we want to see if the generated charges and currents satisfy the continuity equation.  They should because
# $\partial_t j^0 + \partial_x j^1 = -g\partial_t \partial_x \phi + g \partial_x \partial_t \phi = 0$
# by the fact that partial derivatives commute.  We can plot each of these terms separately to figure out if they would cancel out.

# In[86]:


print(numpy.shape(charge_record))
print(numpy.shape(current_record))


# In[93]:


#Time derivative of charge
timederiv_record = []
for x in range(J): #looping over x
    timedepcharge = charge_record[:,x] #finding time dependence at a particular x
    timederiv = numpy.gradient(timedepcharge,dt)
    timederiv_record.append(timederiv)
timederiv_record = numpy.array(timederiv_record)

#X derivative of current
xderiv_record = [] 
for t in range(N): #looping over t
    xdepcurrent = current_record[:,t] #finding x dependence at a particular t
    xderiv = numpy.gradient(xdepcurrent,dx)
    xderiv_record.append(xderiv)
xderiv_record = numpy.array(xderiv_record)

print(numpy.shape(timederiv_record))
print(numpy.shape(xderiv_record))

for t in range(N):
    #pyplot.plot(x_grid, timederiv_record[:,t], label='charge')
    #pyplot.plot(x_grid, xderiv_record[t], label='current')
    pyplot.plot(x_grid, numpy.add(timederiv_record[:,t],xderiv_record[t]), label='continuity equation')
    pyplot.xlim(-150,150)
    pyplot.legend()
    pyplot.show()


# Checking to see if KG equation is satisfied

# In[94]:


numpy.shape(U_record[7])


# In[118]:


LHS = numpy.add(g*g*U_record, g*extfield)

xderivterm_record = []
for t in range(N): #loop over times
    firstxderiv = numpy.gradient(U_record[t],dx) #take derivative of x component at specified t
    secondxderiv = numpy.gradient(firstxderiv,dx)
    xderivterm_record.append(secondxderiv)
xderivterm_record = numpy.array(xderivterm_record)

tderivterm_record = []
for x in range(J): #loop over x
    firsttderiv = numpy.gradient(U_record[:,x],dt) #take derivative of t component at specified x
    secondtderiv = numpy.gradient(firsttderiv,dt)
    tderivterm_record.append(secondtderiv)
tderivterm_record = numpy.array(tderivterm_record)

print(numpy.shape(xderivterm_record))
print(numpy.shape(tderivterm_record))

RHS = numpy.subtract(tderivterm_record, numpy.transpose(xderivterm_record))


# In[120]:


for t in range(N):
    #pyplot.plot(x_grid, LHS[t], label='efield')
    #pyplot.plot(x_grid, RHS[:,t], label='derivatives')
    pyplot.plot(x_grid, numpy.add(LHS[t], RHS[:,t]), label="KG equation")
    pyplot.xlim(-150,150)
    pyplot.legend()
    pyplot.show()


# In[122]:


NewLHS = LHS[1:N-1]
NewLHS = NewLHS[:,1:J-1]
tderivterm_record = []
for t in range(N-2):
    tderivterm_record.append( (U_record[t+1+1][1:J-1]-2*U_record[t+1][1:J-1]+U_record[t+1-1][1:J-1])/(dt**2) )
tderivterm_record = numpy.array(tderivterm_record)
xderivterm_record = []
for x in range(J-2):
    secondxderiv = []
    for t in range(N-2):
        secondxderiv.append( (U_record[t+1][x+1+1]-2*U_record[t+1][x+1]+U_record[t+1][x+1-1] + U_record[t+1+1][x+1+1]-2*U_record[t+1+1][x+1]+U_record[t+1+1][x+1-1])/(2*dx**2) )
    xderivterm_record.append(secondxderiv)
xderivterm_record = numpy.array(xderivterm_record)
print(numpy.shape(xderivterm_record))
print(numpy.shape(tderivterm_record))
NewRHS = numpy.subtract(numpy.transpose(tderivterm_record), xderivterm_record)


# In[126]:


for t in range(N-2):
    pyplot.plot(x_grid[1:J-1], NewLHS[t], label='efield')
    pyplot.plot(x_grid[1:J-1], NewRHS[:,t], label='derivatives')
    pyplot.plot(x_grid[1:J-1], numpy.add(NewLHS[t], NewRHS[:,t]), label="KG equation")
    pyplot.xlim(-150,150)
    pyplot.legend()
    pyplot.show()


# In[ ]:




