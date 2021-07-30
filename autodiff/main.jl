#!/usr/bin/env julia

include("header.jl")
include("pepsUnitCell.jl")
include("computeIsometries.jl")
include("absorptions.jl")
include("CTMRG.jl")
include("fixedpoint.jl")
include("expectationValues.jl")

0; # suppress REPL output

Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];

chiB = 3;
chiE = 5;
truncBelowE = 1e-6;
d = 2;

latticeTens = Dict{Tuple{Int, Int}, Array{Float64, 5}}();
for idx = 1 : Lx, idy = 1 : Ly
    push!(latticeTens, (idx, idy) => randn(chiB, chiB, d, chiB, chiB));
end
# push!(latticeTens, (1, 1) => randn(3, 5, d, 4, 6));
# push!(latticeTens, (2, 1) => randn(4, 6, d, 3, 5));
iPEPS = pepsUnitCell(Lx, Ly, latticeTens, unitCellLayout);

initMethod = 0;
convTol = 1e-8;
maxIter = 100;
CTMRGTensors = runCTMRG(iPEPS, chiE, truncBelowE, convTol, maxIter, initMethod);

σ₀ = [1 0; 0  1]
σ₁ = [0 1; 1  0]
σ₃ = [1 0; 0 -1]
σ₂ = -1im*σ₃*σ₁

function isingTBG(h, J; id=0.0)
    reshape(h*(kron(σ₃, σ₀) + kron(σ₀, σ₃)) + J*kron(σ₁, σ₁) + id*kron(σ₀, σ₀), (2,2,2,2))
end

tbg = isingTBG(1., 1., id=0.0)

energy(iPEPS, CTMRGTensors, tbg)