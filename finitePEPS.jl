
using Arpack
using IterativeSolvers
using LinearMaps
using TensorKit: ⊗


include("models/getSpinOperators.jl")
include("computeEnvironmentFinitePEPS.jl")
include("computeEnvironmentFinitePEPS_PEPO.jl")
include("expectationValues_finitePEPS.jl")
include("environments_finitePEPS.jl")

# clear console
Base.run(`clear`)

# # SU2 example
# v1 = SU2Space(0 => 1, 1 => 1);
# v2 = SU2Space(0 => 1, 1 => 1);
# T = TensorMap(randn, v1 ⊗ v1, v2 ⊗ v2)
# print(T)

# define types for PEPS tensors
const PEPSType{S} = AbstractTensorMap{S,3,2} where {S<:EuclideanSpace}
const RedPEPSType{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}

include("constructPEPO.jl")
include("parameters.jl")
include("engines/PEPS_contractions.jl")

@time P = generateParameters()
finitePEPO = constructPEPOIsing(P)
finitePEPO = constructPEPOIdentity(P)

physicalSpin = P["spin"];
d = Int(2 * physicalSpin + 1);
vecSpacePhys = ℂ^d;
vecSpaceTriv = ℂ^1;
vecSpaceVirt = ℂ^2;
# PEPSTensor = TensorMap(randn, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^3, ℂ^4 ⊗ ℂ^5);

# initialize PEPS network
Lx = P["Lx"];
Ly = P["Ly"];
finitePEPS = initializeFinitePEPS(Lx, Ly, vecSpacePhys, vecSpaceVirt, vecSpaceTriv);

# # compute reduced tensors
# redFinitePEPS = computeReducedTensors(finitePEPS);

# set environment bond dimension
chiE = P["χ"][1];
@time envTensors_NORM = computeEnvironmentFinitePEPS_NORM(finitePEPS, finitePEPS, chiE);
@time envTensors_PEPO = computeEnvironmentFinitePEPS_PEPO(finitePEPS, finitePEPO, chiE);


pepsTensor = finitePEPS[1,1]
pepoTensor = finitePEPO[1,1]
pepsTensorSize = dim(pepsTensor)

applyPEPO_OneSiteLM = LinearMap(pepsTensorSize) do pepsTensor
    return applyPEPO_OneSite(pepsTensor, pepoTensor, envTensors_PEPO)
end

applyNORM_OneSiteLM = LinearMap(pepsTensorSize) do pepsTensor
    return applyNORM_OneSite(pepsTensor, envTensors_NORM)
end

pepsTensor = applyNORM_OneSite(pepsTensor, envTensors_NORM[1,1]);
pepsTensor = permute(pepsTensor, (1,2,3), (4,5));
pepsTensor = applyPEPO_OneSite(pepsTensor, pepoTensor, envTensors_PEPO[1,1]);
pepsTensor = permute(pepsTensor, (1,2,3), (4,5));

# r = lobpcg(applyNORM_OneSiteLM, false, 1)
# println(r.λ, r.X)

# # println(envTensors[1,2])

# checkContractions = 2;
# for idx = 1 : Lx, idy = 1 : Ly
#     # if checkContractions == 1
#         expVal_N = computeSingleSiteExpVal(finitePEPS[idx,idy], envTensors_NORM[idx,idy]);
#         println(expVal_N)
#     # elseif checkContractions == 2
#         expVal_P = computeSingleSiteExpVal_PEPO(finitePEPS[idx,idy], finitePEPO[idx,idy], envTensors_PEPO[idx,idy]);
#         println(expVal_P)
#     # end
# end

# 0;

# fA = LinearMap(10; issymmetric=true, isposdef=true) do x
#     return A*x
# end

# fB = LinearMap(10; issymmetric=true, isposdef=true) do x
#     return B*x
# end

# mapNORM = LinearMap

# x₀ = rand(10)
# # @time λ,ϕ = eigs(fA, fB, nev = 1, which = :LM, v0 = x₀)

# r = lobpcg(fA, fB, false, 1)
# println(r.λ, r.X)
# @time λ,ϕ = geneigsolve(x₀, 1, :LM; ishermitian = true, isposdef = true) do x f(x,A,B) end
# λ,ϕ = KrylovKit.geneigsolve(x₀, 1, :LM) do x f(x) end,