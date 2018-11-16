from __future__ import division
import numpy
#import scipy.integrate
from scipy import integrate
import matplotlib.pyplot as pyplot

#define constants for SHM

vdispSHM = 156     #velocity dispersion ms-1

vlagSHM = 230      #earth speed wrt galactic halo, taking into account motion of sun and earth, ms-1


#define dark disc parameters

vdispDD = 50     #velocity dispersion of dark disc in ms-1

vlagDD = 60      #vlag for dark disc, ms-1

NoPoint = 100

# calculate VMin, bound on integral of mean inverse speed
# first define the constants used

#SET-UP FOR NEWS-G NEON DETECTOR

A = 73       #A as in nucleon number of detector nuclei, element ting
Mn = A * 1.6605E-27     #detector nuclear mass - NEON
Mx =100 * 1.79E-27    #WIMP mass 10 GeV   - both in kg

Mxn = (Mx*Mn)/(Mx+Mn)   #reduced WIMP-nucleus mass

Mp = 1.67E-27   #Mass of proton
Mxp = (Mx*Mp)/(Mx+Mp)   #WIMP-NUCLEON reduced mass, so with an individual proton rather than the whole thing


#recast Mx and Mxn in natural units
MxNatural = Mx / 1.79E-27
MxnNatural = Mxn / 1.79E-27

#define recoil energy range we're interested in
ErKeV = numpy.linspace(0.1, 500, NoPoint)   #in KeV for axis labels, then convert to joules for calculations
Er = ErKeV * 1.60218E-16
#print(Er)

Vmin = ((Mn * Er)/(2 * Mxn**2))**0.5    #in ms-1
Vmin = Vmin / 1000  #convert to kms-1


# define constants etc. used in calculating the detect rate
expTime = 365    #exposure time of experiment, days, needs to be days for conversion factor to be right
Mt = 300          #Total detector mass, kg
Eexp = expTime * Mt     #exposure parameter used in calculating detection rate

Rho0 = 0.3      #local DM density, in units of Gev cm-3 but probs have to convert to standard ones

CrossSection = 1E-9       #WIMP-Nucleon scattering cross section, in units of pb, conversion factor to cm2 is 1E-36
CrossSectionSI = (Mxn / Mxp)**2 * A**2 * CrossSection     #Spin Independant Cross section between WIMP-nucleus
#print(CrossSectionSI)

qMom = numpy.arange(NoPoint, dtype=float)
qMom = (2*Mn*Er)**0.5 / 5.26E-19      #momentum transfer, converted to GeV


R = 1.2 * A**0.5        #in fm, may need to convert
s = 1               #nuclear skin thickness, in fm may need to convert
R1 = (R**2 - 5*s**2)**0.5

#calculate j1 sph bessel function used in nuclear form factor

j1 = numpy.arange(NoPoint, dtype=float)
Nff = numpy.arange(NoPoint, dtype=float)
for i in range(NoPoint):
    j1[i] = ((numpy.sin(qMom[i]*R1))/(qMom[i]*R1)**2) - ((numpy.cos(qMom[i]*R1))/(qMom[i]*R1))
    Nff[i] = (3 * j1[i]) / (qMom[i] * R1) * numpy.exp(-0.5 * qMom[i] * s)


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

'''
# calculate and populate arrays for both 1D, 3D for SHM and DD contributions
for i in range(NoPoint):
    VDist1DSHM[i] = ((4*numpy.pi)/(V[i]*vlagSHM*(2*numpy.pi)**1.5*vdispSHM)) * numpy.exp(-1*(V[i]**2 + vlagSHM**2) / (2*vdispSHM**2)) * numpy.sinh((V[i]*vlagSHM)/(vdispSHM**2))
    VDist3DSHM[i] = V[i]**2 * VDist1DSHM[i]

    VDist1DDD[i] = ((4*numpy.pi)/(V[i]*vlagDD*(2*numpy.pi)**1.5*vdispDD)) * numpy.exp(-1*(V[i]**2 + vlagDD**2) / (2*vdispDD**2)) * numpy.sinh((V[i]*vlagDD)/(vdispDD**2))
    VDist3DDD[i] = V[i]**2 * VDist1DDD[i]

    VDist1DSum[i] = 0.5 * VDist1DSHM[i] + 0.5 * VDist1DDD[i]
    VDist3DSum[i] = 0.5 * VDist3DSHM[i] + 0.5 * VDist3DDD[i]
'''

#function to calculate MIS integral
def intgradSHM(V):
    #try the other formulation from dm101 that isnt as direct
    #print((V * vlagSHM) / vdispSHM ** 2)
    res = (2**0.5/((numpy.pi)**0.5*vdispSHM*vlagSHM)) * numpy.exp(-1*((V**2 + vlagSHM**2)/(2*vdispSHM**2))) * numpy.sinh((V*vlagSHM)/vdispSHM**2)
    return res

def intgradDD(V):

    #try the other formulation from dm101 that isnt as direct
    res = (2**0.5/((numpy.pi)**0.5*vdispDD*vlagDD)) * numpy.exp(-1*((V**2 + vlagDD**2)/(2*vdispDD**2))) * numpy.sinh((V*vlagDD)/vdispDD**2)
    return res

# calculate mean inverse speed integral

def intsolSHM(Vmin):
    res = integrate.quad(intgradSHM, Vmin, 1000)[0]
    return res

def intsolDD(Vmin):
    res = integrate.quad(intgradDD, Vmin, 4000)[0]
    return res

MISSHM = numpy.arange(NoPoint, dtype=float)
MISDD = numpy.arange(NoPoint, dtype=float)
MISSum = numpy.arange(NoPoint, dtype=float)

for i in range(NoPoint):
    MISSHM[i] = intsolSHM(Vmin[i])
    MISDD[i] = intsolDD(Vmin[i])

for i in range(NoPoint):
    MISSum[i] = 0.5 * MISSHM[i] + 0.5 * MISDD[i]

#calculate the detection rate from MIS and Nff etc
#need to set up detectrate things as arrays
DetectRateSHM = numpy.arange(NoPoint, dtype=float)
DetectRateDD = numpy.arange(NoPoint, dtype=float)
DetectRateSum = numpy.arange(NoPoint, dtype=float)


#conversion factor constant to get to the 'right' units
#ConvFactor = 1.609E-10 / 100  #this ones the wrong one from when i thought it was unitless
ConvFactor = 4.35E5
#print(ConvFactor)

for i in range(NoPoint):

    DetectRateSHM[i] = Eexp * (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSectionSI * Nff[i] * MISSHM[i] * ConvFactor
    DetectRateDD[i] = Eexp * (Rho0 / (2 * MxNatural * MxnNatural**2)) * CrossSectionSI * Nff[i] * MISDD[i] * ConvFactor
    DetectRateSum[i] = Eexp * (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSectionSI * Nff[i] * MISSum[i] * ConvFactor

    #DetectRateSHM[i] = DetectRateSHM[i] / Eexp

    #DetectRateSHM[i] = (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSHM[i] * ConvFactor
    #DetectRateDD[i] = (Rho0 / (2 * MxNatural * MxnNatural**2)) * CrossSection * Nff[i] * MISDD[i] * ConvFactor
    #DetectRateSum[i] = (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSum[i] * ConvFactor

    #DetectRateSHM[i] = DetectRateSHM[i] / Eexp
    #DetectRateDD[i] = DetectRateDD[i] / Eexp
    #DetectRateSum[i] = DetectRateSum[i] / Eexp

#print(DetectRateSHM)

def EventIntegral(ErKeVInt):
    ErInt = ErKeVInt * 1.60218E-16
    qMomInt = (2 * Mn * ErInt) ** 0.5 / 5.26E-19
    #print(qMomInt)
    j1Int = ((numpy.sin(qMomInt*R1))/(qMomInt*R1)**2) - ((numpy.cos(qMomInt*R1))/(qMomInt*R1))
    NffInt = (3 * j1Int) / (qMomInt * R1) * numpy.exp(-0.5 * qMomInt * s)
    VminInt = ((Mn * ErInt) / (2 * Mxn ** 2)) ** 0.5
    VminInt = VminInt / 1000
    MISSHMInt = intsolSHM(VminInt)
    #print(MISSHMInt)
    res = Eexp * (Rho0 / (2 * MxNatural * MxnNatural ** 2)) * CrossSectionSI * NffInt * MISSHMInt* ConvFactor
    #print(res)
    return res

#Calculate total number of detections by integrating the rate over the energy
Ethreshold = 10
Emax = 100
EventTot = integrate.quad(EventIntegral, Ethreshold, Emax)[0]
print(EventTot)

pyplot.figure(figsize=[10,10])


#plot the 3 subplots
'''
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
'''

#pyplot.plot(V, DetectRateSHM)
#pyplot.plot(V, DetectRateSum)

pyplot.plot(ErKeV, DetectRateSHM, label='SHM')
pyplot.plot(ErKeV, DetectRateDD, label='DD')
pyplot.plot(ErKeV, DetectRateSum, label='SHM+DD')

#pyplot.plot(Er, j1)
#pyplot.plot(ErKeV, Nff**2)

pyplot.legend()

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.ylim(1E-1, 1E2)

pyplot.xlabel(r'$Recoil Energy (KeV)$')
pyplot.ylabel(r'$Detection Rate (KeV ^{-1})$')

pyplot.show()




