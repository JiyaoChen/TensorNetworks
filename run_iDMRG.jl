
push!(LOAD_PATH, pwd())

using DMRG_engine
using DMRG_types
using TensorKit
using LinearAlgebra
using Combinatorics

# include models
include("getSpinOperators.jl")
include("models/mpoIsing.jl")
include("models/mpoHeisenberg.jl")

# clear console
Base.run(`clear`)

# simulation parameters
χ = 250
tol = 1e-16
numSteps = 1000

# choose model
setModel = 2
if setModel == 1
    setSym = "Z2"
    J = 4.0
    h = 2.0
    mpo = mpoIsing(J = J, h = h, setSym = setSym)
elseif setModel == 2
    setSym = "SU2"
    J = 1.0
    spinS = 1/2
    mpo = mpoHeisenberg(J = J, spinS = spinS, setSym = setSym)
end

@time mps = DMRG_engine.iDMRG2(mpo, χ=χ, numSteps=numSteps, tol=tol)


# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=6, setSym="")

# length = 10
# mps_arr = Vector([tens for tens in psi for i in 1:length/2])

# mps = DMRG_types.MPS(mps_arr);

0;


# spinS = 1/2;
# vP = SU₂Space(spinS => 1);
# mpoSpaceL = SU₂Space(0 => 2, 1 => 1);
# mpoSpaceR = SU₂Space(0 => 2, 1 => 1);
# γ = sqrt(spinS * (spinS + 1));

# # construct empty MPO
# Sx, Sy, Sz, Id = getSpinOperators(spinS);
# ham_arr = zeros(ComplexF64, dim(vP), dim(mpoSpaceL), dim(mpoSpaceR), dim(vP));
# ham_arr[:,1,3,:] = J * Sx;
# ham_arr[:,1,4,:] = J * Sy;
# ham_arr[:,1,5,:] = J * Sz;
# ham_arr[:,3,2,:] = Sx;
# ham_arr[:,4,2,:] = Sy;
# ham_arr[:,5,2,:] = Sz;
# ham_arr = reshape(ham_arr,10,10)
# # # print(blocks(mpo))

# countJ = 0


# # construct empty MPO
# mpo = TensorMap(zeros, ComplexF64, vP ⊗ mpoSpaceL, mpoSpaceR ⊗ vP)

# # fill tensor blocks
# tensorDict = convert(Dict, mpo);
# dictKeys = keys(tensorDict);
# dataDict = tensorDict[:data];

# matrix = Array{ComplexF64}([1.0 0.0 0.0 ; 0.0 1.0 J * γ ; γ 0.0 0.0])
# # for perm in permutations(matrix)
# #     dataDict["Irrep[SU₂](1/2)"] = perm
# #     tensorDict = convert(Dict, mpo)
# #     mpo = convert(TensorMap, tensorDict)

# # end
