import json
import yaml
from yaml.loader import SafeLoader
import numpy as np
import ufl
import dolfinx

class Var_from_yml():
    def __init__(self,yml_file: str) -> None:
        with open(yml_file, 'r') as file:
            self.inputs = yaml.load(file,Loader=SafeLoader)
            
    def derived_constants(self):
        Mech = self.inputs["Mechanical"]
        Hydr = self.inputs["Hydraulic"]
        Therm = self.inputs["Thermal"]
        Plas = self.inputs["Plastic"]
        
        Mech["K"] = Mech["E"]/3/(1-2*Mech["nu"])
        Mech["G"] = Mech["E"]/2/(1+Mech["nu"])
        Mech["alpha"] = 1 - Mech["K"]/Mech["Ks"]
        Mech["M"] = (Hydr["phi"]/Mech["Kf"] + (Mech["alpha"]-Hydr["phi"])/Mech["Ks"])**(-1)
        Mech["Ku"] = Mech["K"] + Mech["alpha"]**2 * Mech["M"]
        Mech["B"] = (1-Mech["K"]/Mech["Ku"])/Mech["alpha"]
        Hydr["kappa"] = Hydr["k"]/Hydr["mu"]

        Therm["rho"] = Hydr["phi"]*Hydr["rhof"] + (1-Hydr["phi"])*Mech["rhos"]
        Therm["Cp"] = Hydr["phi"]*Therm["Cpf"] + (1-Hydr["phi"])*Therm["Cps"]
        Therm["k"] = Hydr["phi"]*Therm["kf"] + (1-Hydr["phi"])*Therm["ks"]
        Therm["alpha"] = Therm["k"]/(Therm["rho"]*Therm["Cp"])
        Therm["betae"] = Mech["alpha"]*Therm["betas"] + Hydr["phi"]*(Therm["betaf"]-Therm["betas"])

        Mech["rhob"] = Therm["rho"]
        
        Plas["softening"] = bool(Plas["softening"])
        Plas["phi"] *= np.pi/180
        Plas["phi_crit"] *= np.pi/180
        Plas["psi"] *= np.pi/180

        Plas["M"] = 6*np.sin(Plas["phi"])/(3-np.sin(Plas["phi"]))
        Plas["M_crit"] = 6*np.sin(Plas["phi_crit"])/(3-np.sin(Plas["phi_crit"]))
        Plas["Mg"] = 6*np.sin(Plas["psi"])/(3-np.sin(Plas["psi"]))
        Plas["q"] = (1+np.sin(Plas["phi"]))/(1-np.sin(Plas["phi"]))
        Plas["So"] = Plas["UCS"]/2/np.sqrt(Plas["q"])
        Plas["c"] = 6*Plas["So"]*np.cos(Plas["phi"])/(3-np.sin(Plas["phi"]))

        try:
            Plas["UTS"]
        except:
            Plas["UTS"] = Plas["UCS"]/20
        return self.inputs

class dict_to_class():
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def return_class(i):
    var = json.loads(json.dumps(i), object_hook=lambda d: dict_to_class(**d))
    return var.Mechanical,var.Hydraulic,var.Thermal,var.Plastic

class Elastoplasticity():
    def __init__(self,name,mesh,functions,old_functions,init_conditions):
        assert type(name) == str, "file name must be a string"
        assert type(mesh) == dolfinx.mesh.Mesh, "mesh must be a dolfinx.mesh"
        for j in functions:
            if functions[j] is not None:
                assert type(functions[j]) == dolfinx.fem.function.Function, f"{j} must be a dolfinx.fem.Function"

        for k in old_functions:
            if old_functions[k] is not None:
                assert type(old_functions[k]) == dolfinx.fem.function.Function, f"old {k} must be a dolfinx.fem.Function"
                
        for l in init_conditions:
            if init_conditions[l] is not None:
                assert type(init_conditions[l]) == dolfinx.fem.function.Function, f"{l} init. condition must be a dolfinx.fem.Function"
        
        self.name = name
        self.mesh = mesh
        self.eps = np.finfo(dtype=np.float64).eps
        
        self.u = functions['displacement']
        self.εp = functions['plastic_strain']
        self.p = functions['pressure']
        self.q = functions['flux']
        self.T = functions['temperature']
        η = functions['mobilized_friction']
        self.η = ufl.variable(η) if η is not None else η
        
        self.u_n = old_functions['displacement']
        self.εp_n = old_functions['plastic_strain']
        self.p_n = old_functions['pressure']
        self.T_n = old_functions['temperature']
        
        self.p_i = init_conditions['pressure']
        self.T_i = init_conditions['temperature']
        
        properties = Var_from_yml(f"{name}.yml").derived_constants()
        Mech,Hydr,Therm,Plas = return_class(properties)
        
        self.dim = ufl.shape(functions['displacement'])[0]
        self.K = dolfinx.fem.Constant(mesh,Mech.K)
        self.G = dolfinx.fem.Constant(mesh,Mech.G)
        self.α = dolfinx.fem.Constant(mesh,Mech.alpha)
        self.αL = dolfinx.fem.Constant(mesh,Therm.betas/3)
        
        self.M_peak = dolfinx.fem.Constant(mesh,Plas.M)
        self.M_cs = dolfinx.fem.Constant(mesh,Plas.M_crit)
        self.a_cs = dolfinx.fem.Constant(mesh,Plas.a)
        self.M_psi = dolfinx.fem.Constant(mesh,Plas.Mg)
        self.cohes = dolfinx.fem.Constant(mesh,Plas.c)
        
        self.ρf = dolfinx.fem.Constant(mesh,Hydr.rhof)
        self.Mb = dolfinx.fem.Constant(mesh,Mech.M)
        self.βe = dolfinx.fem.Constant(mesh,Therm.betae)
        self.κ = dolfinx.fem.Constant(mesh,Hydr.kappa)
        
        self.Cv = dolfinx.fem.Constant(mesh,Therm.rho*Therm.Cp)
        self.Cvf = dolfinx.fem.Constant(mesh,Hydr.rhof*Therm.Cpf)
        self.kT = dolfinx.fem.Constant(mesh,Therm.k)
        
        self.zero = dolfinx.fem.Constant(mesh,0.0)
        self.zero_array = ufl.as_tensor(np.zeros((self.dim,self.dim)))
        
    def total_strain(self):
        return ufl.sym(ufl.grad(self.u))
    
    def elastic_strain(self):
        εe = self.total_strain() - self.εp
        εe -= ufl.Identity(self.dim)*self.αL*(self.T-self.T_i) if self.T is not None else self.zero_array
        return εe
    
    def volumetric_strain(self):
        return ufl.nabla_div(self.u)
    
    def delta_volumetric_strain(self):
        return ufl.nabla_div(self.u) - ufl.nabla_div(self.u_n)
    
    def elastic_deviator_strain(self):
        ε = self.total_strain()
        ε_q = ufl.sqrt((2/3)*ufl.inner(ufl.dev(ε),ufl.dev(ε)))
        return ufl.conditional(ε_q < self.eps, 0.0, ε_q)
    
    def plastic_deviator_strain(self):
        εp_q_ = ufl.sqrt((2/3)*ufl.inner(ufl.dev(self.εp),ufl.dev(self.εp)))
        εp_q = ufl.conditional(εp_q_ < self.eps, 0.0, εp_q_)
        return ufl.variable(εp_q) if self.η is not None else εp_q
    
    def trial_elastic_strain(self):
        εe_n = self.total_strain() - self.εp_n
        εe_n -= ufl.Identity(self.dim)*self.αL*(self.T-self.T_i) if self.T is not None else self.zero_array
        return εe_n
        
    def total_stress(self):
        εe = self.elastic_strain()
        σ = (self.K-2*self.G/3)*ufl.tr(εe)*ufl.Identity(self.dim) + 2*self.G*εe
        σ -= self.α*self.p*ufl.Identity(self.dim) if self.p is not None else self.zero_array
        return σ
    
    def effective_stress(self):
        σ = self.total_stress()
        σ += self.p*ufl.Identity(self.dim) if self.p is not None else self.zero_array
        self.effective_stress_variable(σ)
        return σ
    
    def effective_stress_variable(self,σ):
        self.σ_eff = ufl.variable(σ)
    
    def trial_effective_stress(self):
        εe_n = self.trial_elastic_strain()
        σ = (self.K-2*self.G/3)*ufl.tr(εe_n)*ufl.Identity(self.dim) + 2*self.G*εe_n
        σ += (1-self.α)*self.p*ufl.Identity(self.dim) if self.p is not None else self.zero_array
        return σ
   
    def invariants(self,trial=False):
        σ = self.σ_eff if trial == False else self.trial_effective_stress()
        q_ = ufl.sqrt((3/2)*ufl.inner(ufl.dev(σ),ufl.dev(σ)))
        q = ufl.conditional(q_ < self.eps, 0.0, q_)
        p = (1/3)*ufl.tr(σ)
        return p,q
        
    def yield_surface(self):
        p,q = self.invariants()
        M = self.η if self.η is not None else self.M_peak
        return q + M*p - self.cohes

    def plastic_potential(self):
        p,q = self.invariants()
        M = self.η-self.M_cs if self.η is not None else self.M_psi
        return q + M*p

    def plastic_normals(self):
        dfdσ = ufl.diff(self.yield_surface(),self.σ_eff)
        dgdσ = ufl.diff(self.plastic_potential(),self.σ_eff)   
        return dfdσ, dgdσ

    def softening_modulus(self):
        f = self.yield_surface()
        εp_q = self.plastic_deviator_strain()
        return -ufl.diff(f,self.η)*ufl.diff(self.η,εp_q) if self.η is not None else self.zero
    
    def plastic_modulus(self):
        H = self.softening_modulus()
        dgdp = self.η-self.M_cs if self.η is not None else self.M_psi
        dfdp = self.η if self.η is not None else self.M_peak
        return 3*self.G + self.K*dgdp*dfdp + H + self.eps

    def plastic_multiplier(self):
        p,q = self.invariants(trial=True)
        M = self.η if self.η is not None else self.M_peak
        f = q + M*p - self.cohes
        f_pos = ufl.max_value(f,0.0)
        dλ = f_pos/self.plastic_modulus()
        return dλ
    
    def mobilized_friction(self,active):
        εp_q = self.plastic_deviator_strain()
        η = self.M_peak - (self.M_peak-self.M_cs)*(1.0-ufl.exp(-self.a_cs*εp_q))
        return self.η - active*η if self.η is not None else self.M_peak
    
    def plastic_strain(self,active):
        _, dgdσ = self.plastic_normals()
        return -(self.εp-self.εp_n) + active*self.plastic_multiplier()*dgdσ
    
    def plastic_consistency(self):
        return self.plastic_multiplier()*self.yield_surface()
        
    def mass_balance(self,Δt,strain_active):
        dεv = self.delta_volumetric_strain()
        eq = (1/self.Mb)*(self.p-self.p_n) + strain_active*self.α*dεv + Δt*ufl.div(self.q)
        if self.T is not None:
            eq -= self.βe*(self.T - self.T_n)
        return self.ρf*eq
    
    def darcy_flux(self,Δt,δq):
        eq = (1/self.κ)*ufl.inner(self.q,δq) - self.p*ufl.div(δq)
        return Δt*eq
    
    def energy_balance(self,Δt,δT,advection=False,f=False):
        eq = (self.Cv/Δt)*(self.T-self.T_n)*δT + self.kT*ufl.inner(ufl.grad(self.T),ufl.grad(δT))
        if advection:
            # SUPG: Brooks and Hughes, 1982, Comput. Methods Appl. Mech. Eng.
            r = (self.Cv/Δt)*(self.T-self.T_n) + self.Cvf*ufl.inner(self.q,ufl.grad(self.T)) - self.kT*ufl.div(ufl.grad(self.T))
            if f:
                r -= f
                eq -= f*δT
            tau = ufl.CellDiameter(self.mesh)/2/ufl.sqrt(ufl.dot(self.q,self.q)+self.eps)
            eq += self.Cvf*ufl.inner(self.q,ufl.grad(self.T))*δT + tau*ufl.dot(self.q,ufl.grad(δT))*r
        return eq
    
    def plastic_consistency(self):
        return self.yield_surface()*self.plastic_multiplier()