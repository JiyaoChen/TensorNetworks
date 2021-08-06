#!/usr/bin/env julia

include("header.jl")
include("customAdjoints.jl")
include("pepsUnitCell.jl")
include("computeIsometries.jl")
include("absorptions.jl")
include("CTMRG.jl")
include("fixedpoint.jl")
include("expectationValues.jl")

# iPEPS settings
Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];

chiB = 2;
d = 2;

# CTMRG settings
initMethod = 0;
convTol = 1e-8;
maxIter = 10;
chiE = 4;
truncBelowE = 1e-6;

# initialize iPEPS tensors
pepsTensors = Array{Array{Float64, 5}, 2}(undef, Lx, Ly);
for idx = 1 : Lx, idy = 1 : Ly
    pepsTensors[idx, idy] = randn(chiB, chiB, d, chiB, chiB);
end
# pepsTensors[1, 1] = randn(3, 5, d, 4, 6);
# pepsTensors[2, 1] = randn(4, 6, d, 3, 5);

# # initialize structs for CTMRG tensors
# initMethod = 0;
# CTMRGTensors = initializeCTMRGTensors(pepsTensors, unitCellLayout, chiE, initMethod = initMethod);

σ₀ = [1.0 0.0; 0.0  1.0]
σ₁ = [0.0 1.0; 1.0  0.0]
σ₃ = [1.0 0.0; 0.0 -1.0]
σ₂ = -1im*σ₃*σ₁

function isingTBG(h, J; id = 0.0)
    reshape(h*(kron(σ₃, σ₀) + kron(σ₀, σ₃)) + J*kron(σ₁, σ₁) + id*kron(σ₀, σ₀), (2,2,2,2))
end

function heisenbergTBG(Jx, Jy, Jz, h; id = 0.0)
    reshape(Jx*kron(σ₁, σ₁) + Jy*kron(σ₂, σ₂) + Jz*kron(σ₃, σ₃) + h*(kron(σ₃, σ₀) + kron(σ₀, σ₃)) + id*kron(σ₀, σ₀), (2, 2, 2, 2))
end

# energyTBG = isingTBG(0.5, 1., id=0.0)
energyTBG = heisenbergTBG(1.0, 1.0, 1.0, 0.0)


function computeEnergy(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)

    # reshape pepsTensor into array of iPEPS tensors
    Lx = size(pepsTensorsVec, 1);
    Ly = size(pepsTensorsVec, 2);
    pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] for idx = 1 : Lx, idy = 1 : Ly];

    # run CTMRG
    CTMRGTensors = runCTMRG(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod);

    # compute energy
    return energy(pepsTensors, unitCellLayout, CTMRGTensors, energyTBG)

end

# CTMRGTensors, back = Zygote.pullback(CTMRGStep, CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE));
# 0;

# function optimisePEPS(iPEPS,  energyTBG; χ::Int, tol::Real, maxit::Int, optimargs = (), optimmethod = LBFGS(m = 20))
    # let energy = x -> real(computeEnergy(iPEPS, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG))
    #     res = optimize(energy, Δ -> Zygote.gradient(energy,Δ)[1], bulk, optimmethod, inplace = false, optimargs...)
    # end
# end

function optimizePEPS(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)
   
    optimmethod = LBFGS(m = 20);
    optimargs = (Optim.Options(f_tol = 1e-6, show_trace = true), );
    let energy = x -> real(computeEnergy(x, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG))
        res = optimize(energy, Δ -> Zygote.gradient(energy, Δ)[1], pepsTensors, optimmethod, inplace = false, optimargs...);
    end

    return res;

end

pepsTensors = rand(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);
minPEPS = optimizePEPS(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)