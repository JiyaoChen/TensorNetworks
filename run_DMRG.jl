
push!(LOAD_PATH, pwd())

using DMRG_types
using DMRG_engine

# # include models
include("getSpinOperators.jl")
# include("models/mpoIsing.jl")
# include("models/mpoHeisenberg.jl")
include("models/mpoHeisenbergU1.jl")
include("parameters.jl")
include("initialVS.jl")

# clear console
Base.run(`clear`)

parameters = generateParameters()
hamiltonian = generateHeisenbergU1(parameters)
initialVectorSpaces = generateInitialVS(parameters)

model = DMRG_types.Model(hamiltonian, initialVectorSpaces, parameters)

mps = DMRG_types.MPS(model, init = ones)
env = DMRG_types.MPOEnvironments(mps, model.H)

# just checking if the contractions work
@tensor env.mpoEnvL[1][1 2 4] * mps.ACs[1][4 5 6] * hamiltonian.mpo[1][2 3 8 5] * conj(mps.ACs[1][1 3 7]) * env.mpoEnvR[1][6 8 7]
# checking if the edge Hamiltonians contain the right terms (J ZZ + [J/2 Sp Sm + h.c.])
@tensor tbgate[-1 -2 -3; -4 -5 -6] := hamiltonian.mpo[1][-1 -2 1 -6] * hamiltonian.mpo[end][1 -3 -4 -5]
tbgateArr = reshape(convert(Array, tbgate), (dim(codomain(tbgate)),dim(domain(tbgate))))
nzIndices = findall(x->x!=0, tbgateArr)
[tbgateArr[nz] for nz in nzIndices]