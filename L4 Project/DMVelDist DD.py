from __future__ import division
import numpy
#import scipy.integrate
from scipy import integrate
import matplotlib.pyplot as pyplot

#define constants for SHM

vdispSHM = 156000     #velocity dispersion kms-1

vlagSHM = 230000      #earth speed wrt galactic halo, taking into account motion of sun and earth, kms-1


#define dark disc parameters

vdispDD = 50000     #velocity dispersion of dark disc in kms-1

vlagDD = 60000      #vlag for dark disc, kms-1


# define x axis velocity base
#vvals = numpy.linspace(0, 1000, 1000)

NoPoint = 100

# calculate VMin, bound on integral of mean inverse speed
# first define the constants used

Mn = 2.17E-25     #detector nuclear mass  - this is for XENON1T CURRENTLY
Mx = 8.95E-26      #WIMP mass   - both in kg

Mxn = (Mx*Mn)/(Mx+Mn)   #reduced WIMP-nucleon mass

Er = 3.2E-16      #recoil energy in joules, == 2kev

Vmin = ((Mn * Er)/(2 * Mxn**2))**0.5

print(Vmin)


#calculate 1D and 3D velocity distribution

VDist1DSHM = numpy.arange(NoPoint, dtype=float)
VDist3DSHM = numpy.arange(NoPoint, dtype=float)

VDist1DDD = numpy.arange(NoPoint, dtype=float)
VDist3DDD = numpy.arange(NoPoint, dtype=float)

VDist1DSum = numpy.arange(NoPoint, dtype=float)
VDist3DSum = numpy.arange(NoPoint, dtype=float)

V = numpy.linspace(1E-6, 1000, NoPoint)
V = V*1000
#print(V)

# calculate and populate arrays for both 1D, 3D for SHM and DD contributions
for i in range(NoPoint):
    #VDist1DSHM[i] = (1/((2*numpy.pi)**1.5*vdispSHM**3)) * numpy.exp((-1*(V[i]-vlagSHM)**2)/(2*vdispSHM**2))
    VDist1DSHM[i] = ((4*numpy.pi)/(V[i]*vlagSHM*(2*numpy.pi)**1.5*vdispSHM)) * numpy.exp(-1*(V[i]**2 + vlagSHM**2) / (2*vdispSHM**2)) * numpy.sinh((V[i]*vlagSHM)/(vdispSHM**2))
    VDist3DSHM[i] = V[i]**2 * VDist1DSHM[i]

    #print(VDist1DSHM[i])

    #VDist1DDD[i] = (1/((2*numpy.pi)**1.5*vdispDD**3)) * numpy.exp((-1*(V[i]-vlagDD)**2)/(2*vdispDD**2))
    VDist1DDD[i] = ((4*numpy.pi)/(V[i]*vlagDD*(2*numpy.pi)**1.5*vdispDD)) * numpy.exp(-1*(V[i]**2 + vlagDD**2) / (2*vdispDD**2)) * numpy.sinh((V[i]*vlagDD)/(vdispDD**2))
    VDist3DDD[i] = V[i]**2 * VDist1DDD[i]

    #print(VDist1DDD[i])

    VDist1DSum[i] = 0.5 * VDist1DSHM[i] + 0.5 * VDist1DDD[i]
    VDist3DSum[i] = 0.5 * VDist3DSHM[i] + 0.5 * VDist3DDD[i]

    #print(VDist1DSum[i])

#function to calculate MIS integral
def intgradSHM(V):

    #try the other formulation from dm101 that isnt as direct
    res = (2**0.5/((numpy.pi)**0.5*vdispSHM*vlagSHM)) * numpy.exp(-1*((V**2 + vlagSHM**2)/(2*vdispSHM**2))) * numpy.sinh((V*vlagSHM)/vdispSHM**2)

    #return VDist1D / V
    return res

def intgradDD(V):

    #try the other formulation from dm101 that isnt as direct
    res = (2**0.5/((numpy.pi)**0.5*vdispDD*vlagDD)) * numpy.exp(-1*((V**2 + vlagDD**2)/(2*vdispDD**2))) * numpy.sinh((V*vlagDD)/vdispDD**2)

    #return VDist1D / V
    return res

# calculate mean inverse speed integral

def intsolSHM(V):
    res = integrate.quad(intgradSHM, Vmin, V)[0]
    #print(res)
    return res

def intsolDD(V):
    res = integrate.quad(intgradDD, Vmin, V)[0]
    #print(res)
    return res

MISSHM = numpy.arange(NoPoint, dtype=float)
MISDD = numpy.arange(NoPoint, dtype=float)
MISSum = numpy.arange(NoPoint, dtype=float)
for i in range(NoPoint):
    MISSHM[i] = intsolSHM(V[i])
    MISDD[i] = intsolDD(V[i])

for i in range(NoPoint):
    MISSum[i] = 0.5 * MISSHM[i] + 0.5 * MISDD[i]

#flip MIS round so its correct, problem with the calculation method
MISSHM = numpy.amax(MISSHM) - MISSHM
MISSum = numpy.amax(MISSum) - MISSum

pyplot.figure(figsize=[10,10])

#plot the 3 subplots
pyplot.subplot(311)
pyplot.plot(V, VDist1DSHM * 1E17)
pyplot.plot(V, VDist1DSum * 1E17)
pyplot.ylabel(r'$f(v) / 10 ^{-6} km ^{-3} s ^{3}$')


pyplot.subplot(312)
pyplot.plot(V, VDist3DSHM * 1E6)
pyplot.plot(V, VDist3DSum * 1E6)
pyplot.ylabel(r'$f _{3}(v) / 10 ^{-3} km ^{-3} s ^{3}$')

pyplot.subplot (313)
pyplot.plot(V, MISSHM * 1E6)
pyplot.plot(V, MISSum * 1E6)
pyplot.xlabel(r'$v / (kms ^{-1})$')
pyplot.ylabel(r'$\eta (v) / 10 ^{-3} km ^{-1} s$')

pyplot.show()




