
push!(LOAD_PATH, pwd())

using DMRG_types
using DMRG_engine

# # include models
include("getSpinOperators.jl")
# include("models/mpoIsing.jl")
# include("models/mpoHeisenberg.jl")
include("models/mpoHeisenbergSU2.jl")
include("models/mpoHeisenbergU1.jl")
include("models/mpoHeisenbergNoSym.jl")
include("models/mpoIsingNoSym.jl")
include("parameters.jl")
include("initialVS.jl")

# clear console
Base.run(`clear`)

### SU2 story
@time parameters = generateParameters()
@time hamiltonian = generateHeisenbergSU2(parameters)
@time initialVectorSpaces = generateInitialVSSU2(parameters)

@time model = DMRG_types.Model(hamiltonian, initialVectorSpaces, parameters)

@time mps = DMRG_types.MPS(model, init = ones)
@time env = DMRG_types.MPOEnvironments(mps, model.H)

@time mps = DMRG_engine.DMRG2(mps, env, model)

### U1 story
@time parameters = generateParameters()
@time hamiltonian = generateHeisenbergU1(parameters)
@time initialVectorSpaces = generateInitialVS(parameters)

@time model = DMRG_types.Model(hamiltonian, initialVectorSpaces, parameters)

@time mps = DMRG_types.MPS(model, init = ones)
@time env = DMRG_types.MPOEnvironments(mps, model.H)

@time mps = DMRG_engine.DMRG2(mps, env, model)

### NoSym story
@time parameters = generateParameters()
@time hamiltonian = generateHeisenbergNoSym(parameters)
@time initialVectorSpaces = generateInitialVSNoSym(parameters)

@time model = DMRG_types.Model(hamiltonian, initialVectorSpaces, parameters)

@time mps = DMRG_types.MPS(model, init = ones)
@time env = DMRG_types.MPOEnvironments(mps, model.H)

@time mps = DMRG_engine.DMRG2(mps, env, model)


# @time parameters = generateParameters()
# @time hamiltonian = generateIsingNoSym(parameters)
# @time initialVectorSpaces = generateInitialVSNoSym(parameters)

# @time model = DMRG_types.Model(hamiltonian, initialVectorSpaces, parameters)

# @time mps = DMRG_types.MPS(model, init = ones)
# @time env = DMRG_types.MPOEnvironments(mps, model.H)

# @time mps = DMRG_engine.DMRG2(mps, env, model)

# just checking if the contractions work
# @tensor env.mpoEnvL[1][1 2 4] * mps.ACs[1][4 5 6] * hamiltonian.mpo[1][2 3 8 5] * conj(mps.ACs[1][1 3 7]) * env.mpoEnvR[1][6 8 7]
# checking if the edge Hamiltonians contain the right terms (J ZZ + [J/2 Sp Sm + h.c.])
# bVL = Tensor(ones, space(hamiltonian.mpo[1], 1)')
# bVR = Tensor(ones, space(hamiltonian.mpo[end], 3)')
# @tensor tbgate[-1 -2 -3; -4 -5 -6] := hamiltonian.mpo[1][-1 -2 1 -5] * hamiltonian.mpo[end][1 -3 -4 -6]
# @tensor tbgate_reduced[-1 -2; -3 -4] := bVL[1] * tbgate[1 -1 -2 2 -3 -4] * bVR[2]
# @tensor tbgate[-1 -2 -3 -4; -5 -6 -7 -8] := hamiltonian.mpo[1][-1 -2 1 -6] * hamiltonian.mpo[2][1 -3 2 -7] * hamiltonian.mpo[end][2 -4 -5 -8]
# @tensor tbgate_reduced[-1 -2 -3; -4 -5 -6] := bVL[1] * tbgate[1 -1 -2 -3 2 -4 -5 -6] * bVR[2]

# tbgateArr = reshape(convert(Array, tbgate_reduced), (dim(codomain(tbgate_reduced)),dim(domain(tbgate_reduced))))
# nzIndices = findall(x->x!=0, tbgateArr)
# [tbgateArr[nz] for nz in nzIndices]
0;