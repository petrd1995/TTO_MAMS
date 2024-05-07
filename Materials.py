
class Steel:
    E = 2.00e5  
    G = 7.69e4  
    Y = 3.45e8  
    k = 50 
    alpha = 12e-6 
    densitymm = 7.85e-9  
    density = 7850 

class Inconel718:
    E = 1.93e11 
    G = 7.46e4
    Y = 1.24e9
    k = 6.5
    alpha = 13e-6
    density = 8900

class Inconel718mm:
    E = 1.93e5
    G = 7.46e4
    Y = 1.24e9
    k = 6.5
    alpha = 13e-6
    density = 8900*1e-9

class Aluminum2024:
    E = 0.724e5
    G = 2.70e4
    Y = 4.05e8 
    k = 180
    alpha = 23e-6
    density = 2700

class Aluminum6061:
    E = 0.689e5
    G = 2.55e4
    Y = 3.5e8
    k = 167
    alpha = 23.6e-6
    density = 2700


class Titanium:
    E = 1.45e5
    G = 5.38e4
    Y = 8.62e8
    k = 25
    alpha = 11e-6
    density = 4500

class Water:
    def __init__(self, Tw_in, radius, velocity):
        self.Tw_in = Tw_in
        self.diameter = radius*2
        self.velocity = velocity
    @property
    def Cp(self):
        return 12010.1471-80.4072879*self.Tw_in**1+0.309866854*self.Tw_in**2-5.38186884E-4*self.Tw_in**3+3.62536437E-7*self.Tw_in**4
    @property
    def k(self):
        return -0.869083936+0.00894880345*self.Tw_in**1-1.58366345E-5*self.Tw_in**2+7.97543259E-9*self.Tw_in**3
    @property
    def dyn(self):
        return 1.3799566804-0.021224019151*self.Tw_in**1+1.3604562827E-4*self.Tw_in**2-4.6454090319E-7*self.Tw_in**3+8.9042735735E-10*self.Tw_in**4-9.0790692686E-13*self.Tw_in**5+3.8457331488E-16*self.Tw_in**6

    @property
    def rho(self):
        if self.Tw_in<=293.15:
            return 0.000063092789034*self.Tw_in**3-0.060367639882855*self.Tw_in**2+18.9229382407066*self.Tw_in-950.704055329848
            
        elif self.Tw_in>293.15:
            return 0.000010335053319*self.Tw_in**3-0.013395065634452*self.Tw_in**2+4.969288832655160*self.Tw_in+432.257114008512

    @property
    def kin(self):
        return self.dyn/self.rho

    @property
    def Pr(self):
        return self.dyn*self.Cp/self.k
    
    @property
    def Re(self):
        return self.rho*self.diameter*self.velocity/self.dyn

    @property
    def Nu(self, pipe='temperature', Re=None, Pr=None):
        if Re is not None:
            self.Re = Re
        if Pr is not None:
            self.Pr = Pr

        if pipe=='flux'and self.Re<2300 :
            self.flow_type = 'laminar'
            return 4.36
        elif pipe=='temperature' and self.Re<2300 :
            self.flow_type = 'laminar'
            return 3.66
        elif self.Re>2300: 
            self.flow_type = 'turbulent'
            return 0.023*self.Re**0.8*self.Pr**0.4


class Dummy:
    E = 1
    G = 1
    Y = 1
    k = 1
    alpha = 1
    density = 1