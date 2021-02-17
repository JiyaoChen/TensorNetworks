push!(LOAD_PATH, pwd())
using DMRG_engine
using DMRG_types
using TensorKit
using LinearAlgebra

# clear console
Base.run(`clear`)

# simulation parameters
χ = 10
tol = 1e-6
steps = 10
setSym = "Z2"
J = 4
h = 2

# create the MPO => put in new object!!!
if setSym == ""
    vP = ℂ^2;
    vV = ℂ^1;
    vL = ℂ^3;
    vR = ℂ^3;
elseif setSym == "Z2"
    vP = ℤ₂Space(0 => 1, 1 => 1);
    vV = ℤ₂Space(0 => 1);
    vL = ℤ₂Space(0 => 2, 1 => 1);
    vR = ℤ₂Space(0 => 2, 1 => 1);
end

# Pauli opertators
Id = Matrix{ComplexF64}(I,2,2);
X = 1/2 * [ 0 +1 ; +1 0 ];
Y = 1/2 * [ 0 -1im ; +1im 0 ];
Z = 1/2 * [ +1 0 ; 0 -1 ];

# generate the Ising MPO
ham_arr = zeros(Complex{Float64}, dim(vP), dim(vL), dim(vR), dim(vP));
ham_arr[:,1,1,:] = Id;
ham_arr[:,2,2,:] = Id;
ham_arr[:,1,3,:] = -J*X;
ham_arr[:,3,2,:] = X;
ham_arr[:,1,2,:] = -h*Z;
mpo = TensorMap(ham_arr, vP*vL, vR*vP)

@time psi = DMRG_engine.iDMRG2(mpo, χ=χ, steps=steps, tol=tol)
# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=6, setSym="")

# length = 10
# mps_arr = Vector([tens for tens in psi for i in 1:length/2])

# mps = DMRG_types.MPS(mps_arr);