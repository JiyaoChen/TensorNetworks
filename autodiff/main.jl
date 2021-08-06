#!/usr/bin/env julia

# set project directory
if ~any(occursin.(pwd() * "/autodiff", LOAD_PATH))
    push!(LOAD_PATH, pwd() * "/autodiff")
end

# clear console
Base.run(`clear`)

using vPEPS
using Profile

# include required functions
include("models.jl")

# iPEPS settings
# Lx = 2;
# Ly = 2;
# unitCellLayout = [1 3 ; 2 4];

Lx = 1;
Ly = 1;
unitCellLayout = [1 1 ; 1 1];

chiB = 2;
d = 2;

# CTMRG settings
initMethod = 0;
convTol = 1e-8;
maxIter = 10;
chiE = 4;
truncBelowE = 1e-12;

# # initialize iPEPS tensors
# pepsTensors = Array{Array{Float64, 5}, 2}(undef, Lx, Ly);
# for idx = 1 : Lx, idy = 1 : Ly
#     pepsTensors[idx, idy] = randn(chiB, chiB, d, chiB, chiB);
# end
# pepsTensors[1, 1] = randn(3, 5, d, 4, 6);
# pepsTensors[2, 1] = randn(4, 6, d, 3, 5);

# energyTBG = isingTBG(0.5, 1., id=0.0)
energyTBG = heisenbergTBG(1.0, 1.0, 1.0, 0.0);
energyTBG = ein"aecf, be, fd -> abcd"(energyTBG, σ₁, σ₁');

pepsTensors = rand(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);
@profview minPEPS = optimizePEPS(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)

# final objective value
# Lx = 2, Ly = 2 - (chiB, chiE)
# (2, 4) => -6.624950e-01 

# @profview computeEnergy(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG) # Heisenberg model: −0.6694421(4) [arXiv:1101.3281]
# @time computeEnergy(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG) # for the Heisenberg model: −0.6694421(4) [arXiv:1101.3281]