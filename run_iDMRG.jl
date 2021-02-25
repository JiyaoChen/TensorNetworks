
push!(LOAD_PATH, pwd())

using DMRG_types
using DMRG_engine

using TensorKit
# using LinearAlgebra

# # include models
include("models/getSpinOperators.jl")
include("models/mpoIsing.jl")
include("models/mpoHeisenberg.jl")

include("parameters.jl")
include("initialVS.jl")

# clear console
Base.run(`clear`)

# simulation parameters
χ = 100
tol = 1e-16
systemSize = 100

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
    setSym = "SU2"
    J = 1.0
    spinS = 1/2
    mpo = mpoHeisenberg(J = J, spinS = spinS, setSym = setSym)
end

@time mps = DMRG_engine.iDMRG2(mpo, χ=χ, systemSize=systemSize, tol=tol)
0;