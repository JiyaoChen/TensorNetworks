
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
χ = 100
tol = 1e-16
numSteps = 10

# choose model
setModel = 2
if setModel == 1
    # setSym = ""
    setSym = "Z2"
    J = 4.0
    h = 2.0
    mpo = mpoIsing(J = J, h = h, setSym = setSym)
elseif setModel == 2
    setSym = ""
    setSym = "U1"
    setSym = "SU2"  ## TODO I removed this case in the MPO for now due to some weird issue with the debugger...
    J = -1.0
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