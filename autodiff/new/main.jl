#!/usr/bin/env julia

# set project directory
if ~any(occursin.(pwd(), LOAD_PATH))
    push!(LOAD_PATH, pwd())
end
if ~any(occursin.(pwd() * "/autodiff/new", LOAD_PATH))
    push!(LOAD_PATH, pwd() * "/autodiff/new")
end

# clear console
Base.run(`clear`)

using OMEinsum
using Revise
using vPEPS
using Profile

# include required functions
include("models.jl")

# iPEPS settings

# Lx = 1;
# Ly = 1;
# unitCellLayout = [1 1 ; 1 1];

Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];
# unitCellLayout = reshape(collect(1 : Lx * Ly), Lx, Ly)

# Lx = 2;
# Ly = 2;
# unitCellLayout = reshape(collect(1 : Lx * Ly), Lx, Ly);

chiB = 2;
d = 2;

# CTMRG settings
initMethod = 0;
convTolE = 1e-8;
maxIter = 100;
chiE = 8;
truncBelowE = 1e-8;

# # initialize iPEPS tensors
# pepsTensors = Array{Array{Float64, 5}, 2}(undef, Lx, Ly);
# A = randn(chiB, chiB, d, chiB, chiB);
# for idx = 1 : Lx, idy = 1 : Ly
#     pepsTensors[idx, idy] = randn(chiB, chiB, d, chiB, chiB);
# end

# # Lx = 1, Ly = 2 different chiB
# if Lx == 1 && Ly == 2
#     if all(size(unitCellLayout) .== [Lx, Ly])
#         pepsTensors[1, 1] = randn(2, 3, d, 4, 3);
#         pepsTensors[1, 2] = randn(4, 5, d, 2, 5);
#     else
#         pepsTensors[1, 1] = randn(2, 3, d, 4, 5);
#         pepsTensors[1, 2] = randn(4, 5, d, 2, 3);
#     end
# end

# # Lx = 2, Ly = 1 different chiB
# if Lx == 2 && Ly == 1
#     if all(size(unitCellLayout) .== [Lx, Ly])
#         pepsTensors[1, 1] = randn(2, 3, d, 2, 4);
#         pepsTensors[2, 1] = randn(5, 4, d, 5, 3);
#     else
#         pepsTensors[1, 1] = randn(2, 3, d, 4, 5);
#         pepsTensors[2, 1] = randn(4, 5, d, 2, 3);
#     end
# end

# # # Lx = 2, Ly = 2 different chiB
# if Lx == 2 && Ly == 2
#     pepsTensors[1, 1] = randn(2, 3, d, 4, 5);
#     pepsTensors[2, 1] = randn(9, 5, d, 8, 3);
#     pepsTensors[1, 2] = randn(4, 6, d, 2, 7);
#     pepsTensors[2, 2] = randn(8 ,7, d, 9, 6);
# end

# construct two-body gate to compute energy
# energyTBG = isingTBG(1.0, 0.0, id = 0.0)
# energyTBG = heisenbergTBG(0.0, 0.0, 0.0, 0.0, id=1.0);
energyTBG = heisenbergTBG(1.0, 1.0, 1.0, 0.0, id = 0.0);
# energyTBG = ein"aecf, be, fd -> abcd"(energyTBG, σ₁, σ₁');

initializePEPS = 1; # 0 => randn, 1 => simple update
if initializePEPS == 0

    # initialize pepsTesorsVec
    pepsTensorsVec = randn(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);

elseif initializePEPS == 1

    # run simple update
    convTolB = 1e-6;
    gammaTensors, lambdaTensors, singularValueTensor = simpleUpdate_iPEPS(Lx, Ly, d, chiB, convTolB, energyTBG; verbosePrint = false);

    # convert gammaTensors into pepsTensorsVec
    pepsTensorsVec = zeros(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);
    for idx = 1 : Lx, idy = 1 : Ly
        pepsTensorsVec[idx, idy, :, :, :, :, :] = gammaTensors[idx, idy];
    end

end

# computeEnergy(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)
minPEPS = optimizePEPS(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)

# final objective value
# Lx = 1, Ly = 1 | array columns: chiB, chiE, gsE
energyConv = [2 4 -6.602311e-01];

# final objective value
# Lx = 1, Ly = 1 | array columns: chiB, chiE, gsE
energyConv = [2 4 ];