
push!(LOAD_PATH, pwd())

using DMRG_engine
using DMRG_types
using TensorKit
using LinearAlgebra

# include models
include("getSpinOperators.jl")
include("models/mpoIsing.jl")
include("models/mpoHeisenberg.jl")

# clear console
Base.run(`clear`)

# simulation parameters
χ = 10
tol = 1e-6
steps = 10

# model settings Ising
setSym = "Z2"
J = 1.0
h = 0.5
mpo = mpoIsing(J = J, h = h, setSym = setSym)

# # model settings Heisenberg
# setSym = ""
# J = 1.0
# spinS = 1/2;
# mpo = mpoHeisenberg(J = J, spinS = spinS, setSym = setSym)

@time psi = DMRG_engine.iDMRG2(mpo, χ=χ, steps=steps, tol=tol)
# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
# @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=6, setSym="")

# length = 10
# mps_arr = Vector([tens for tens in psi for i in 1:length/2])

# mps = DMRG_types.MPS(mps_arr);