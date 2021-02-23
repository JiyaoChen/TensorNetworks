
push!(LOAD_PATH, pwd())

using DMRG_types
using DMRG_engine

using TensorKit
# using LinearAlgebra

# # include models
include("getSpinOperators.jl")
# include("models/mpoIsing.jl")
include("models/mpoHeisenberg.jl")
include("models/mpoHeisenbergU1.jl")
include("parameters.jl")
include("initialVS.jl")

# clear console
Base.run(`clear`)

model = DMRG_types.Model(generateHeisenbergU1, generateInitialVS, generateParameters)

mps = DMRG_types.MPS(model, init = ones)
# @time mps = DMRG_engine.DMRG2(mps, model)

# # simulation parameters
# χ = 10
# tol = 1e-16
# size = 6

# # choose model
# setModel = 2
# if setModel == 1
#     # setSym = ""
#     setSym = "Z2"
#     J = 4.0
#     h = 2.0
#     mpo = mpoIsing(J = J, h = h, setSym = setSym)
# elseif setModel == 2
#     setSym = ""
#     setSym = "U1"
#     # setSym = "SU2"
#     J = 1.0
#     spinS = 1/2
#     mpo = mpoHeisenberg(J = J, spinS = spinS, setSym = setSym)
# end

# @time mps = DMRG_engine.iDMRG2(mpo, χ=χ, size=size, tol=tol)
# [println(space(tensor)) for tensor in mps]
# DMRG_types.MPS([tensor for tensor in mps])

# # @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
# # @time DMRG_engine.iDMRG2(χ=χ, maxNumSteps=6, setSym="")


# # length = 10
# # mps_arr = Vector([tens for tens in psi for i in 1:length/2])

# # mps = DMRG_types.MPS(mps_arr);

0;