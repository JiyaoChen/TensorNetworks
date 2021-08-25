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
Lx = 1;
Ly = 1;
unitCellLayout = [1 1 ; 1 1];

Lx = 1;
Ly = 2;
unitCellLayout = [1 2 ; 2 1];
# unitCellLayout = reshape(collect(1 : Lx * Ly), Lx, Ly)

Lx = 2;
Ly = 2;
unitCellLayout = reshape(collect(1 : Lx * Ly), Lx, Ly);

chiB = 3;
d = 2;

# CTMRG settings
initMethod = 0;
convTol = 1e-8;
maxIter = 100;
chiE = 11;
truncBelowE = 1e-8;

# initialize iPEPS tensors
pepsTensors = Array{Array{Float64, 5}, 2}(undef, Lx, Ly);
A = randn(chiB, chiB, d, chiB, chiB);
for idx = 1 : Lx, idy = 1 : Ly
    pepsTensors[idx, idy] = randn(chiB, chiB, d, chiB, chiB);
end

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
energyTBG = isingTBG(0., 0., id = 1.0)
# energyTBG = heisenbergTBG(0.0, 0.0, 0.0, 0.0, id=1.0);
# energyTBG = heisenbergTBG(1.0, 1.0, 1.0, 0.0, id=0.0);
# energyTBG = ein"aecf, be, fd -> abcd"(energyTBG, σ₁, σ₁');

# # initialize pepsTesorsVec
# pepsTensorsVec = rand(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);

# # reshape pepsTensor into array of iPEPS tensors
# Lx = size(pepsTensorsVec, 1);
# Ly = size(pepsTensorsVec, 2);
# pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] for idx = 1 : Lx, idy = 1 : Ly];

# using LinearAlgebra
# using OMEinsum
# using Printf
# using Zygote
# # include("customAdjoints.jl")
# include("CTMRG.jl")
# # include("vPEPS.jl")
# include("vPEPS_methods.jl")

# # get size
# Lx, Ly = size(pepsTensors);

# # get element type and dimensions of pepsTensors
# elementType = eltype(eltype(pepsTensors));
# dimensionsTensors = [size(pepsTensors[idx, idy]) for idx = 1 : Lx, idy = 1 : Ly];

# # initialize structs for CTMRG tensors
# CTMRGTensors = initializeCTMRGTensors(elementType, dimensionsTensors, unitCellLayout, chiE, initMethod = initMethod);

# _, back = Zygote.pullback(CTMRGStep, CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE));
# 0;

computeEnergy(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)
# minPEPS = optimizePEPS(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)

# final objective value
# Lx = 1, Ly = 1
# (2, 4) => -6.602310e-01 / -6.592901e-01
# (2, 8) => 

# Lx = 2, Ly = 2 - (chiB, chiE)
# (2, 4) => -6.624950e-01 
# @profview computeEnergy(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG) # Heisenberg model: −0.6694421(4) [arXiv:1101.3281]
# @time computeEnergy(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG) # for the Heisenberg model: −0.6694421(4) [arXiv:1101.3281]