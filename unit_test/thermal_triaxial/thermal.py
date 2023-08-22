import dolfinx
import dolfiny
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import numpy as np
import pandas as pd
import time

# ----- Custom scripts ----- #
import gmshapi
import poromechanics.constitutive as pm

# ----- Mesh from gmsh ----- #
name, comm = "thermal", MPI.COMM_WORLD
model, gdim, height,elem = gmshapi.mesh_cylindrical_sample()

mesh, mts = dolfiny.mesh.gmsh_to_dolfin(model, gdim, comm)
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mts, gdim - 0)
boundaries, boundaries_keys = dolfiny.mesh.merge_meshtags(mts, gdim - 1)
omega = subdomains_keys["Omega"]
gamma_s = boundaries_keys["Gamma_s"]
gamma_t = boundaries_keys["Gamma_t"]
gamma_b = boundaries_keys["Gamma_b"]

# ----- FEM function spaces ----- #
quad_degree, disp_degree = 2, 2

Vu_e = ufl.VectorElement("CG", mesh.ufl_cell(), disp_degree)
Vεp_e  = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=quad_degree, quad_scheme="default", symmetry=True)
Vp_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
Vq_e = ufl.FiniteElement("RT", mesh.ufl_cell(), 1)
VT_e = ufl.FiniteElement("CG", mesh.ufl_cell(), disp_degree-1)

Vu = dolfinx.fem.FunctionSpace(mesh, Vu_e)
Vεp = dolfinx.fem.FunctionSpace(mesh, Vεp_e)
Vp = dolfinx.fem.FunctionSpace(mesh, Vp_e)
Vq = dolfinx.fem.FunctionSpace(mesh, Vq_e)
VT = dolfinx.fem.FunctionSpace(mesh, VT_e)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": quad_degree})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries, metadata={"quadrature_degree": quad_degree})

# ----- Solution variables ----- #
u = dolfinx.fem.Function(Vu, name="Displacement")
εp = dolfinx.fem.Function(Vεp, name="Plastic_Strain")
p = dolfinx.fem.Function(Vp, name="Pore_Pressure")
q = dolfinx.fem.Function(Vq, name="Darcy_Flux")
T = dolfinx.fem.Function(VT, name="Temperature")

u_n = dolfinx.fem.Function(Vu, name="Displacement0")
εp_n = dolfinx.fem.Function(Vεp, name="Plastic_Strain0")
p_n = dolfinx.fem.Function(Vp, name="Pore_Pressure0")
T_n = dolfinx.fem.Function(VT, name="Temperature0")

p_i = dolfinx.fem.Function(Vp, name="Pore_Pressure_init")
T_i = dolfinx.fem.Function(VT, name="Temperature_init")

δu = ufl.TestFunction(Vu)
δεp = ufl.TestFunction(Vεp)
δp = ufl.TestFunction(Vp)
δq = ufl.TestFunction(Vq)
δT = ufl.TestFunction(VT)
δε = ufl.sym(ufl.grad(δu))

# ----- Constitutive Laws ----- #
functions = {'displacement': u, 'plastic_strain': εp, 'pressure': p,
             'flux': q, 'temperature': T, 'mobilized_friction': None}
old_functions = {'displacement': u_n, 'plastic_strain': εp_n, 'pressure': p_n,
                 'temperature': T_n}
init_conditions = {'pressure': p_i, 'temperature': T_i}

EP = pm.Elastoplasticity(name,mesh,functions,old_functions,init_conditions)

ε = EP.total_strain()
εv = EP.volumetric_strain()
εq = EP.elastic_deviator_strain()
εpq = EP.plastic_deviator_strain()
σ = EP.total_stress()
σ_eff = EP.effective_stress()

# ----- Initialize loading ----- #
load_rate = 0.2778
load_factor = dolfinx.fem.Constant(mesh, 0.0)
plasticity_active = dolfinx.fem.Constant(mesh, 0.0)
strain_active = dolfinx.fem.Constant(mesh, 1.0)

max_temp = 250.0

kk = 25
load = np.linspace(0.0, 1.0, int(kk+1.0))
time_ = load*max_temp/load_rate
dt = time_[1] - time_[0]

# ----- Variational formulation ----- #
pc = dolfinx.fem.Constant(mesh,PETSc.ScalarType(-10.0e6))
n = ufl.FacetNormal(mesh)

F0 = ufl.inner(δε,σ)*dx - ufl.inner(pc*n,δu)*ds(gamma_s)
F1 = ufl.inner(δεp, EP.plastic_strain(plasticity_active))*dx
F2 = ufl.inner(δp,EP.mass_balance(dt,strain_active))*dx
F3 = EP.darcy_flux(dt,δq)*dx
F3 += p_i*dt*ufl.inner(n,δq)*ds(gamma_t) + p_i*dt*ufl.inner(n,δq)*ds(gamma_b)
#F4 = (T-T_n)*δT*dx + dt*Therm.alpha*ufl.inner(ufl.grad(T),ufl.grad(δT))*dx
F4 = EP.energy_balance(dt,δT,advection=False,f=False)*dx

# ----- PETSc solver control ----- #
opts = PETSc.Options(name)
opts["-snes_type"] = "newtonls"
opts["-snes_linesearch_type"] = "basic"
opts["-snes_atol"] = 1.0e-10
opts["-snes_stol"] = 1.0e-2
opts["-snes_rtol"] = 1.0e-12
opts["-snes_max_it"] = 25
opts["-ksp_type"] = "preonly"
opts["-pc_type"] = "lu"
opts["-pc_factor_mat_solver_type"] = "mumps"
opts["-snes_convergence_test"] = "default"
opts["-snes_converged_reason"] = None

problem = dolfiny.snesblockproblem.SNESBlockProblem([F0,F1,F2,F3,F4], [u,εp,p,q,T], prefix=name)

top_dofs_u = dolfiny.mesh.locate_dofs_topological(Vu.sub(2), boundaries, gamma_t)
base_dofs_u = dolfiny.mesh.locate_dofs_topological(Vu.sub(2), boundaries, gamma_b)

top_dofs_T = dolfiny.mesh.locate_dofs_topological(VT, boundaries, gamma_t)
base_dofs_T = dolfiny.mesh.locate_dofs_topological(VT, boundaries, gamma_b)

top_dofs_q = dolfiny.mesh.locate_dofs_topological(Vq, boundaries, gamma_t)
base_dofs_q = dolfiny.mesh.locate_dofs_topological(Vq, boundaries, gamma_b)
side_dofs_q = dolfiny.mesh.locate_dofs_topological(Vq, boundaries, gamma_s)
no_flow = dolfinx.fem.Function(Vq)
no_flow.x.array[:] = 0.0

results = {'σa': [], 'σr': [], 'εa': [], 'εv': [], 'pp': [],'T': [], 't': []}
Volume = dolfiny.expression.assemble(1.0, dx(omega))
Nax = ufl.as_vector([0,0,1])
Nr = ufl.as_vector([np.sqrt(0.5),np.sqrt(0.5),0])

# ----- Loading loop ----- #
start = time.time()
for step, factor in enumerate(load):
    load_factor.value = -factor*max_temp

    problem.bcs = []
    problem.bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(-6.0e-5), top_dofs_u, Vu.sub(2)))
    problem.bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0), base_dofs_u, Vu.sub(2)))
    problem.bcs.append(dolfinx.fem.dirichletbc(load_factor, top_dofs_T, VT))
    problem.bcs.append(dolfinx.fem.dirichletbc(load_factor, base_dofs_T, VT))
    problem.bcs.append(dolfinx.fem.dirichletbc(no_flow,side_dofs_q))
    
    if step > 0:
        strain_active = 1.0
        plasticity_active.value = 1.0

    print('\n \033[1;31m +++ Step: %0.0f of %0.0f +++ \033[0m' %(step,len(load)-1))

    dλ_f = 1
    while dλ_f > 1.0e-1:
        problem.solve()
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"
        dλ_f = dolfiny.expression.assemble(EP.plastic_consistency(),dx(omega))/Volume
    print('\033[1;31m +++ dλ*f = %0.6f +++ \033[0m' %dλ_f)

    results['σa'].append(dolfiny.expression.assemble(ufl.dot(σ_eff*Nax,Nax),dx(omega))/Volume)
    results['σr'].append(dolfiny.expression.assemble(ufl.dot(σ_eff*Nr,Nr),dx(omega))/Volume)
    results['εa'].append(dolfiny.expression.assemble(ufl.dot(ε*Nax,Nax),dx(omega))/Volume)
    results['εv'].append(dolfiny.expression.assemble(εv,dx(omega))/Volume)
    results['pp'].append(dolfiny.expression.assemble(p,dx(omega))/Volume)
    results['T'].append(dolfiny.expression.assemble(T,dx(omega))/Volume)
    results['t'].append(time_[step])

    for source, target in zip([u,εp,p,T], [u_n,εp_n,p_n,T_n]):
        with source.vector.localForm() as locs, target.vector.localForm() as loct:
            locs.copy(loct)

print("\n \033[1;31m Simulation Done! Elapsed time: %0.3f minutes \033[0m" %((time.time()-start)/60))

for key in results:
        results[key].insert(0,0)

σa = np.array(results['σa'])/-1e6
σr = np.array(results['σr'])/-1e6
εa = np.array(results['εa'])*-1e2
εv = np.array(results['εv'])*-1e2
pp = np.array(results['pp'])/1e6
T = np.array(results['T'])
t = np.array(results['t'])/60

dict_ = {'Axial_stress':σa,'Radial_stress':σr,'Axial_strain':εa,'Volumetric_strain':εv,'Pore_pressure':pp,'Temperature':T,'time':t}

df = pd.DataFrame(dict_,dtype=np.float64)
df.to_csv(f"{name}.csv")