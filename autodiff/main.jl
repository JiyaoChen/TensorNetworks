#!/usr/bin/env julia

# set project directory
if ~any(occursin.(pwd(), LOAD_PATH))
    push!(LOAD_PATH, pwd())
end
if ~any(occursin.(pwd() * "/autodiff", LOAD_PATH))
    push!(LOAD_PATH, pwd() * "/autodiff")
end

# clear console
Base.run(`clear`)

using OMEinsum
using Revise
using vPEPS
# using Profile
# using Zygote
# using LinearAlgebra
# using IterTools: imap, iterated

# include required functions
include("models.jl")
# include("CTMRG.jl")
# include("vPEPS_methods.jl")

# iPEPS settings

Lx = 1;
Ly = 1;
unitCellLayout = [1 1 ; 1 1];

Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];

# Lx = 2;
# Ly = 2;
# unitCellLayout = reshape(collect(1 : Lx * Ly), Lx, Ly);

chiB = 3;
d = 2;

# CTMRG settings
initMethod = 1;
convTolE = 1e-10;
maxIter = 50;
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

# # reshape pepsTensor into array of iPEPS tensors
# Lx = size(pepsTensorsVec, 1);
# Ly = size(pepsTensorsVec, 2);
# # pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] for idx = 1 : Lx, idy = 1 : Ly];
# pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] / norm(pepsTensorsVec[idx, idy, :, :, :, :, :]) for idx = 1 : Lx, idy = 1 : Ly];
# normalization = norm(pepsTensorsVec)
# println("Norm of PEPS tensors: $normalization")
# # pepsTensors = pepsTensorsVec;

# # run CTMRG
# CTMRGTensors = runCTMRG(pepsTensors, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod);

# # compute energy
# # gsE = energy(pepsTensors, unitCellLayout, CTMRGTensors, energyTBG);

# CT, back = Zygote.pullback(CTMRGStep, (CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE))...);


# @show computeEnergy(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)
minPEPS = optimizePEPS(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)

# final objective value
# Lx = 1, Ly = 1 | array columns: chiB, chiE, gsE
# energyConv = [2 4 -6.602311e-01; 2 8 -6.602304e-01];

# final objective value
# Lx = 2, Ly = 1 | array columns: chiB, chiE, gsE
# energyConv = [2 2 -6.559741e-01, 2 4 -6.601328e-01, 2 8 -6.562421e-01];


# MC comparison
# E/N = −0.6694421(4)