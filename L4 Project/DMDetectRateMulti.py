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

NoPoint = 1000

# calculate VMin, bound on integral of mean inverse speed
# first define the constants used

#SET-UP FOR NEWS-G NEON DETECTOR

A = 20       #A as in nucleon number of detector nuclei, element ting
Mn = A * 1.6605E-27     #detector nuclear mass - NEON
#Mx =20 * 1.79E-27    #WIMP mass 10 GeV   - both in kg

#setup system to plot multiple WIMP masses at once
Mx1 = 10 * 1.79E-27
Mx2 = 20 * 1.79E-27
Mx3 = 50 * 1.79E-27
Mx4 = 100 * 1.79E-27

#Mxn = (Mx*Mn)/(Mx+Mn)   #reduced WIMP-nucleon mass
Mxn1 = (Mx1*Mn)/(Mx1+Mn)
Mxn2 = (Mx2*Mn)/(Mx2+Mn)
Mxn3 = (Mx3*Mn)/(Mx3+Mn)
Mxn4 = (Mx4*Mn)/(Mx4+Mn)


#recast Mx and Mxn in natural units
#MxNatural = Mx / 1.79E-27
#MxnNatural = Mxn / 1.79E-27

MxNatural1 = Mx1 / 1.79E-27
MxnNatural1 = Mxn1 / 1.79E-27

MxNatural2 = Mx2 / 1.79E-27
MxnNatural2 = Mxn2 / 1.79E-27

MxNatural3 = Mx3 / 1.79E-27
MxnNatural3 = Mxn3 / 1.79E-27

MxNatural4 = Mx4 / 1.79E-27
MxnNatural4 = Mxn4 / 1.79E-27

#define recoil energy range we're interested in
ErKeV = numpy.linspace(0.1, 500, NoPoint)   #in KeV for axis labels, then convert to joules for calculations
Er = ErKeV * 1.60218E-16
#print(ErKeV)

#Vmin = ((Mn * Er)/(2 * Mxn**2))**0.5    #in ms-1
#Vmin = Vmin / 1000  #convert to kms-1

Vmin1 = ((Mn * Er)/(2 * Mxn1**2))**0.5    #in ms-1
Vmin1 = Vmin1 / 1000  #convert to kms-1

Vmin2 = ((Mn * Er)/(2 * Mxn2**2))**0.5    #in ms-1
Vmin2 = Vmin2 / 1000  #convert to kms-1

Vmin3 = ((Mn * Er)/(2 * Mxn3**2))**0.5    #in ms-1
Vmin3 = Vmin3 / 1000  #convert to kms-1

Vmin4 = ((Mn * Er)/(2 * Mxn4**2))**0.5    #in ms-1
Vmin4 = Vmin4 / 1000  #convert to kms-1

#print(Vmin1)
#print(Vmin2)
#print(Vmin3)
#print(Vmin4)

# define constants etc. used in calculating the detect rate
expTime = 7     #exposure time of experiment, days
Mt = 100          #Total detector mass, kg
Eexp = expTime * Mt     #exposure parameter used in calculating detection rate

Rho0 = 0.3      #local DM density, in units of Gev cm-3 but probs have to convert to standard ones

CrossSection = 1       #WIMP-Nucleon scattering cross section, in units of pb, conversion factor to cm2 is 1E-36

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

# calculate and populate arrays for both 1D, 3D for SHM and DD contributions
for i in range(NoPoint):
    VDist1DSHM[i] = ((4*numpy.pi)/(V[i]*vlagSHM*(2*numpy.pi)**1.5*vdispSHM)) * numpy.exp(-1*(V[i]**2 + vlagSHM**2) / (2*vdispSHM**2)) * numpy.sinh((V[i]*vlagSHM)/(vdispSHM**2))
    VDist3DSHM[i] = V[i]**2 * VDist1DSHM[i]

    VDist1DDD[i] = ((4*numpy.pi)/(V[i]*vlagDD*(2*numpy.pi)**1.5*vdispDD)) * numpy.exp(-1*(V[i]**2 + vlagDD**2) / (2*vdispDD**2)) * numpy.sinh((V[i]*vlagDD)/(vdispDD**2))
    VDist3DDD[i] = V[i]**2 * VDist1DDD[i]

    VDist1DSum[i] = 0.5 * VDist1DSHM[i] + 0.5 * VDist1DDD[i]
    VDist3DSum[i] = 0.5 * VDist3DSHM[i] + 0.5 * VDist3DDD[i]


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
    res = integrate.quad(intgradSHM, Vmin, 4000)[0]
    return res

def intsolDD(Vmin):
    res = integrate.quad(intgradDD, Vmin, numpy.inf)[0]
    return res

#MISSHM = numpy.arange(NoPoint, dtype=float)
#MISDD = numpy.arange(NoPoint, dtype=float)
#MISSum = numpy.arange(NoPoint, dtype=float)

MISSHM1 = numpy.arange(NoPoint, dtype=float)
MISDD1 = numpy.arange(NoPoint, dtype=float)
MISSum1 = numpy.arange(NoPoint, dtype=float)

MISSHM2 = numpy.arange(NoPoint, dtype=float)
MISDD2 = numpy.arange(NoPoint, dtype=float)
MISSum2 = numpy.arange(NoPoint, dtype=float)

MISSHM3 = numpy.arange(NoPoint, dtype=float)
MISDD3 = numpy.arange(NoPoint, dtype=float)
MISSum3 = numpy.arange(NoPoint, dtype=float)

MISSHM4 = numpy.arange(NoPoint, dtype=float)
MISDD4 = numpy.arange(NoPoint, dtype=float)
MISSum4 = numpy.arange(NoPoint, dtype=float)
for i in range(NoPoint):
    MISSHM1[i] = intsolSHM(Vmin1[i])
    MISDD1[i] = intsolDD(Vmin1[i])

    MISSHM2[i] = intsolSHM(Vmin2[i])
    MISDD2[i] = intsolDD(Vmin2[i])

    MISSHM3[i] = intsolSHM(Vmin3[i])
    MISDD3[i] = intsolDD(Vmin3[i])

    MISSHM4[i] = intsolSHM(Vmin4[i])
    MISDD4[i] = intsolDD(Vmin4[i])
for i in range(NoPoint):
    MISSum1[i] = 0.5 * MISSHM1[i] + 0.5 * MISDD1[i]
    MISSum2[i] = 0.5 * MISSHM2[i] + 0.5 * MISDD2[i]
    MISSum3[i] = 0.5 * MISSHM3[i] + 0.5 * MISDD3[i]
    MISSum4[i] = 0.5 * MISSHM4[i] + 0.5 * MISDD4[i]

#flip MIS round so its correct, problem with the calculation method
#MISSHM = numpy.amax(MISSHM) - MISSHM
#MISSum = numpy.amax(MISSum) - MISSum

#calculate the detection rate from MIS and Nff etc
#need to set up detectrate things as arrays
DetectRateSHM1 = numpy.arange(NoPoint, dtype=float)
DetectRateDD1 = numpy.arange(NoPoint, dtype=float)
DetectRateSum1 = numpy.arange(NoPoint, dtype=float)

DetectRateSHM2 = numpy.arange(NoPoint, dtype=float)
DetectRateDD2 = numpy.arange(NoPoint, dtype=float)
DetectRateSum2 = numpy.arange(NoPoint, dtype=float)

DetectRateSHM3 = numpy.arange(NoPoint, dtype=float)
DetectRateDD3 = numpy.arange(NoPoint, dtype=float)
DetectRateSum3 = numpy.arange(NoPoint, dtype=float)

DetectRateSHM4 = numpy.arange(NoPoint, dtype=float)
DetectRateDD4 = numpy.arange(NoPoint, dtype=float)
DetectRateSum4 = numpy.arange(NoPoint, dtype=float)

#conversion factor constant to get to the 'right' units
#ConvFactor = 1.609E-10 / 100  #this ones the wrong one from when i thought it was unitless
ConvFactor = 4.35E5
#print(ConvFactor)

for i in range(NoPoint):

    #DetectRateSHM[i] = Eexp * (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSHM[i] * ConvFactor
    #DetectRateDD[i] = Eexp * (Rho0 / (2 * MxNatural * MxnNatural**2)) * CrossSection * Nff[i] * MISDD[i] * ConvFactor
    #DetectRateSum[i] = Eexp * (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSum[i] * ConvFactor

    #DetectRateSHM[i] = (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSHM[i] * ConvFactor
    #DetectRateDD[i] = (Rho0 / (2 * MxNatural * MxnNatural**2)) * CrossSection * Nff[i] * MISDD[i] * ConvFactor
    #DetectRateSum[i] = (Rho0/(2*MxNatural*MxnNatural**2)) * CrossSection * Nff[i] * MISSum[i] * ConvFactor

    #DetectRateSHM[i] = DetectRateSHM[i] / Eexp
    #DetectRateDD[i] = DetectRateDD[i] / Eexp
    #DetectRateSum[i] = DetectRateSum[i] / Eexp


    DetectRateSHM1[i] = Eexp * (Rho0 / (2 * MxNatural1 * MxnNatural1 ** 2)) * CrossSection * Nff[i] * MISSHM1[i] * ConvFactor
    DetectRateDD1[i] = Eexp * (Rho0 / (2 * MxNatural1 * MxnNatural1 ** 2)) * CrossSection * Nff[i] * MISDD1[i] * ConvFactor
    DetectRateSum1[i] = Eexp * (Rho0 / (2 * MxNatural1 * MxnNatural1 ** 2)) * CrossSection * Nff[i] * MISSum1[i] * ConvFactor

    DetectRateSHM2[i] = Eexp * (Rho0 / (2 * MxNatural2 * MxnNatural2 ** 2)) * CrossSection * Nff[i] * MISSHM2[i] * ConvFactor
    DetectRateDD2[i] = Eexp * (Rho0 / (2 * MxNatural2 * MxnNatural2 ** 2)) * CrossSection * Nff[i] * MISDD2[i] * ConvFactor
    DetectRateSum2[i] = Eexp * (Rho0 / (2 * MxNatural2 * MxnNatural2 ** 2)) * CrossSection * Nff[i] * MISSum2[i] * ConvFactor

    DetectRateSHM3[i] = Eexp * (Rho0 / (2 * MxNatural3 * MxnNatural3 ** 2)) * CrossSection * Nff[i] * MISSHM3[i] * ConvFactor
    DetectRateDD3[i] = Eexp * (Rho0 / (2 * MxNatural3 * MxnNatural3 ** 2)) * CrossSection * Nff[i] * MISDD3[i] * ConvFactor
    DetectRateSum3[i] = Eexp * (Rho0 / (2 * MxNatural3 * MxnNatural3 ** 2)) * CrossSection * Nff[i] * MISSum3[i] * ConvFactor

    DetectRateSHM4[i] = Eexp * (Rho0 / (2 * MxNatural4 * MxnNatural4 ** 2)) * CrossSection * Nff[i] * MISSHM4[i] * ConvFactor
    DetectRateDD4[i] = Eexp * (Rho0 / (2 * MxNatural4 * MxnNatural4 ** 2)) * CrossSection * Nff[i] * MISDD4[i] * ConvFactor
    DetectRateSum4[i] = Eexp * (Rho0 / (2 * MxNatural4 * MxnNatural4 ** 2)) * CrossSection * Nff[i] * MISSum4[i] * ConvFactor

#print(DetectRateSHM1)



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

#pyplot.plot(ErKeV, DetectRateSHM)
#pyplot.plot(ErKeV, DetectRateDD)
#pyplot.plot(ErKeV, DetectRateSum)

#pyplot.plot(Er, j1)
#pyplot.plot(ErKeV, Nff**2)

pyplot.plot(ErKeV, DetectRateSHM1, label='10GeV')
pyplot.plot(ErKeV, DetectRateSHM2, label='20GeV')
pyplot.plot(ErKeV, DetectRateSHM3, label='50GeV')
pyplot.plot(ErKeV, DetectRateSHM4, label='100GeV')

pyplot.legend()

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.ylim(1E-1, 1E3)

pyplot.xlabel(r'$Recoil Energy (KeV)$')
pyplot.ylabel(r'$Detection Rate (KeV ^{-1})$')

pyplot.show()




