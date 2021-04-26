
# using MPSKit
using TensorKit


include("models/getSpinOperators.jl")
include("computeEnvironmentFinitePEPS.jl")
include("computeEnvironmentFinitePEPS_PEPO.jl")
include("expectationValues_finitePEPS.jl")
include("initializeFinitePEPS.jl")

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

include("parameters.jl")
include("constructPEPO.jl")

@time P = generateParameters()
finitePEPO = constructPEPOIsing(P)

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
chiE = 100;
# @time envTensors_NORM = computeEnvironmentFinitePEPS(finitePEPS, chiE);
@time envTensors_PEPO = computeEnvironmentFinitePEPS_PEPO(finitePEPS, finitePEPO, chiE);

# println(envTensors[1,2])

checkContractions = 2;
for idx = 1 : Lx, idy = 1 : Ly
    if checkContractions == 1
        expVal_N = computeSingleSiteExpVal(finitePEPS[idx,idy], envTensors[idx,idy]);
        println(expVal_N)
    elseif checkContractions == 2
        expVal_P = computeSingleSiteExpVal_PEPO(finitePEPS[idx,idy], finitePEPO[idx,idy], envTensors[idx,idy]);
        println(expVal_P)
    end
end

# for idx = 1, idy = 3
#     println(finitePEPS[idx,idy],"\n")
#     envT = envTensors[idx,idy];
#     for ide = 1 : length(envT)
#         println(envT[ide],"\n")
#     end
# end

0;