#!/usr/bin/env julia

# set project directory
if ~any(occursin.(pwd(), LOAD_PATH))
    push!(LOAD_PATH, pwd())
end

# clear console
Base.run(`clear`)

# load packages
using LinearAlgebra
using OMEinsum
using Optim
using Printf
using Zygote

mutable struct pepsUnitCell{T}
    Lx::Int64
    Ly::Int64
    tensorDict::Dict{Tuple{Int, Int}, T}
    unitCellLayout::Matrix{Int64}
end

function getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout)
    
    # set ordering of tensors in the Lx × Ly unit cell
    tensorNumbers = reshape(collect(1 : Lx * Ly), Lx, Ly)
    
    # convert (latticeIdx, latticeIdy) to (unitCellIdx, unitCellIdy)
    unitCellLx, unitCellLy = size(unitCellLayout);
    unitCellIdx = mod(latticeIdx - 1, unitCellLx) + 1;
    unitCellIdy = mod(latticeIdy - 1, unitCellLy) + 1;

    # get number of tensor
    tensorNum = unitCellLayout[unitCellIdx, unitCellIdy];

    # get posX and posY of (latticeIdx, latticeIdy) in unit cell
    tensorIdx = findfirst(tensorNumbers .== tensorNum);
    posX = tensorIdx[1];
    posY = tensorIdx[2];
    return posX, posY;

end

Base.similar(UC::pepsUnitCell) = pepsUnitCell(UC.Lx, UC.Ly, typeof(UC.tensorDict)(), UC.unitCellLayout)

function Base.getindex(UC::pepsUnitCell, latticeIdx::T, latticeIdy::T) where T <: Integer
    Lx = UC.Lx;
    Ly = UC.Ly;
    tensorDict = UC.tensorDict;
    unitCellLayout = UC.unitCellLayout;
    posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
    return tensorDict[(posX, posY)]
end

function Base.setindex!(UC::pepsUnitCell, pepsTensor, latticeIdx::T, latticeIdy::T) where T <: Integer
    Lx = UC.Lx;
    Ly = UC.Ly;
    tensorDict = UC.tensorDict;
    unitCellLayout = UC.unitCellLayout;
    posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
    tensorDict[(posX, posY)] = pepsTensor;
    return UC
end

function initializeC(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        c = ones(eltype(pepsTensor), 1, 1);
    elseif initMethod == 1
        c = randn(eltype(pepsTensor), χ, χ);
    end
    return c
end

function initializeT1(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T1 = ones(eltype(pepsTensor), 1, size(pepsTensor, 5), size(pepsTensor, 5), 1);
    elseif initMethod == 1
        T1 = randn(eltype(pepsTensor), χ, size(pepsTensor, 5), size(pepsTensor, 5), χ);
    end
    return T1
end

function initializeT2(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T2 = ones(eltype(pepsTensor), size(pepsTensor, 4), size(pepsTensor, 4), 1, 1);
    elseif initMethod == 1
        T2 = randn(eltype(pepsTensor), size(pepsTensor, 4), size(pepsTensor, 4), χ, χ);
    end
    return T2
end

function initializeT3(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T3 = ones(eltype(pepsTensor), 1, 1, size(pepsTensor, 2), size(pepsTensor, 2));
    elseif initMethod == 1
        T3 = randn(eltype(pepsTensor), χ, χ, size(pepsTensor, 2), size(pepsTensor, 2));
    end
    return T3
end

function initializeT4(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T4 = ones(eltype(pepsTensor), 1, size(pepsTensor, 1), size(pepsTensor, 1), 1);
    elseif initMethod == 1
        T4 = randn(eltype(pepsTensor), χ, size(pepsTensor, 1), size(pepsTensor, 1), χ);
    end
    return T4
end


function initializeTensors(initFunc, tensorDict::Dict, χ, initMethod::Int, T, Lx, Ly, unitCellLayout)
    
    # initialize empty struct for unitCell
    unitCell = pepsUnitCell(Lx, Ly, Dict{Tuple{Int, Int}, T}(), unitCellLayout);

    # call initFunc for each element in the tensorDict Dict and return unitCell
    foreach(u -> unitCell[u...] = initFunc(tensorDict[u], χ, initMethod), keys(tensorDict));
    return unitCell

end

function initializeCTMRGTensors(iPEPS::pepsUnitCell, chiE::Int; initMethod = 0)
    
    # get struct variables
    Lx = iPEPS.Lx;
    Ly = iPEPS.Ly;
    tensorDict = iPEPS.tensorDict;
    unitCellLayout = iPEPS.unitCellLayout;

    # set types for C and T tensors
    typeC = Array{eltype(eltype(values(tensorDict))), 2};
    typeT = Array{eltype(eltype(values(tensorDict))), 4};

    C1 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T1 = initializeTensors(initializeT1, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C2 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T2 = initializeTensors(initializeT2, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C3 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T3 = initializeTensors(initializeT3, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C4 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T4 = initializeTensors(initializeT4, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    # return CTMRG tensors
    return C1, T1, C2, T2, C3, T3, C4, T4

end

function runCTMRG(iPEPS, chiE, truncBelowE, initMethod = 0)

    # initialize structs for CTMRG tensors
    CTMRGTensors = initializeCTMRGTensors(iPEPS, chiE, initMethod = initMethod);

    # get physical dimension
    d = size(iPEPS[1,1], 3);

    # initialize variables to control CTMRG procedure
    oldvals = fill(Inf, 4 * chiE * d);
    tol = 0;
    maxIter = 10;

    # construct struct for stopFunction
    stopFunc = StopFunction(oldvals, -1, tol, maxIter)

    # make CTMRG step and return CRMTG tensors
    CTMRGTensors = CTMRGStep((CTMRGTensors..., oldvals), (iPEPS, chiE, truncBelowE, d))[1 : end - 1];
    return CTMRGTensors

end

function CTMRGStep((C1, T1, C2, T2, C3, T3, C4, T4, oldvals), (iPEPS, chiE, truncBelowE, d))
    C4, T4, C1 = absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE)
    # C1, T1, C2 = absorptionStep_U(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE)
    # C2, T2, C3 = absorptionStep_R(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE)
    # C3, T3, C4 = absorptionStep_D(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE)

    return (C1, C2, C3, C4, T1, T2, T3, T4, oldvals)
end

function absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE)
    
    projsUL, projsDL = similar.((T1, T3));
    projsUR, projsDR = similar.((T3, T1));
    nC4, nT4, nC1 = similar.((C4, T4, C1));

    foreach(keys(iPEPS.tensorDict)) do tensorKey
        projsUL[tensorKey...], projsDL[tensorKey...], projsUR[tensorKey...], projsDR[tensorKey...] = computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE, tensorKey);
    end
    foreach(keys(iPEPS.tensorDict)) do tensorKey
        nC4[tensorKey...], nT4[tensorKey...], nC1[tensorKey...] = absorption_L(C1, T1, T3, C4, T4, iPEPS, projsUL, projsDL, tensorKey);
    end

    return nC4, nT4, nC1
end

function computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE, (idX, idY))
    
    # compute upper and lower half of the network
    rhoU, rhoD = horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, (idX, idY))
    d = size(iPEPS[1,1], 3);

    # perform SVD of rhoU
    sizeRhoU = size(rhoU);
    rhoU = reshape(rhoU, prod(sizeRhoU[1 : 3]), prod(sizeRhoU[4 : 6]));
    rhoU /= norm(rhoU);        
    UU, SU, VU = svd(rhoU);
    SU = diagm(SU);
    VU = VU';

    # truncate singular values
    newChi = sum(diag(SU) .> truncBelowE);
    UU = UU[:, 1 : newChi];
    SU = SU[1 : newChi, 1 : newChi];
    VU = VU[1 : newChi, :];

    # absorb sqrt(SU) into UU and VU
    FUL = UU * sqrt(SU);
    FUR = sqrt(SU) * VU;

    # reshape FUL and FUR
    FUL = reshape(FUL, sizeRhoU[1], sizeRhoU[2], sizeRhoU[3], size(FUL, 2));
    FUR = reshape(FUR, size(FUR, 1), sizeRhoU[4], sizeRhoU[5], sizeRhoU[6]);


    # perform SVD of rhoD
    sizeRhoD = size(rhoD);
    rhoD = reshape(rhoD, prod(sizeRhoD[1 : 3]), prod(sizeRhoD[4 : 6]));
    rhoD /= norm(rhoD);
    UD, SD, VD = svd(rhoD);
    SD = diagm(SD);
    VD = VD';

    # truncate singular values
    newChi = sum(diag(SD) .> truncBelowE);
    UD = UD[:, 1 : newChi];
    SD = SD[1 : newChi, 1 : newChi];
    VD = VD[1 : newChi, :];

    # absorb sqrt(SD) into UD and VD
    FDR = UD * sqrt(SD);
    FDL = sqrt(SD) * VD;

    # reshape FDR and FUR
    FDR = reshape(FDR, sizeRhoD[1], sizeRhoD[2], sizeRhoD[3], size(FDR, 2));
    FDL = reshape(FUR, size(FDL, 1), sizeRhoD[4], sizeRhoD[5], sizeRhoD[6]);


    # compute biorthogonalization of FUL and FDL tensors and truncate UL, SL and VL
    @ein BOL[-1, -2] := FDL[-1, 1, 2, 3] * FUL[1, 2, 3, -2];
    BOL /= norm(BOL);
    UL, SL, VL = svd(BOL);
    SL = diagm(SL);
    VL = VL';

    # truncate UL, SL and VL
    newChi = min(chiE, sum(diag(SL) .> truncBelowE));
    UL = UL[:, 1 : newChi];
    SL = SL[1 : newChi, 1 : newChi];
    VL = VL[1 : newChi, :];
    sqrtSL = sqrt(pinv(SL));

    # build projectors for left truncation
    @ein PUL[-1, -2, -3, -4] := FUL[-1, -2, -3, 1] * adjoint(VL)[1, 2] * sqrtSL[2, -4];
    @ein PDL[-1, -2, -3, -4] := FDL[1, -2, -3, -4] * adjoint(UL)[2, 1] * sqrtSL[2, -1];


    # compute biorthogonalization of FUR and FDR tensors
    @ein BOR[-1, -2] := FUR[-1, 1, 2, 3] * FDR[1, 2, 3, -2];
    BOR /= norm(BOR);
    UR, SR, VR = svd(BOR);
    SR = diagm(SR);
    VR = VR';

    # truncate UR, SR and VR
    newChi = min(chiE, sum(diag(SR) .> truncBelowE));
    UR = UR[:, 1 : newChi];
    SR = SR[1 : newChi, 1 : newChi];
    VR = VR[1 : newChi, :];
    sqrtSR = sqrt(pinv(SR));

    # build projectors for right truncation
    @ein PUR[-1, -2, -3, -4] := sqrtSR[-1, 1] * adjoint(UR)[1, 2] * FUR[2, -2, -3, -4];
    @ein PDR[-1, -2, -3, -4] := sqrtSR[-4, 2] * adjoint(VR)[1, 2] * FDR[-1, -2, -3, 1];


    # return projectors
    return  PUL, PDL, PUR, PDR;

end

function absorption_L!(nC1, nT1, nT3, nC4, nT4, C1, T1, T3, C4, T4, iPEPS, projsUL, projsDL, (x, y))
    
    nC1[x + 0, y + 1] = ein"adef, (dc, cefb) -> ab"(projsDL[x - 1, y + 0], C1[x + 0, y + 0], T1[x + 0, y + 0]);

    nT4[x + 0, y + 1] = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[x + 0, y + 0], T4[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), projsUL[x - 1, y + 0], iPEPS[x + 0, y + 0]);
    nC4[x + 0, y + 1] = ein"(cd, cafe), defb -> ab"(C4[x + 0, y + 0], T3[x + 0, y + 0], projsUL[x + 0, y + 0]);

end

function horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, (x, y))

    rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[x + 0, y + 0], C1[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), T1[x + 0, y + 0], iPEPS[x + 0, y + 0]);
    rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[x + 0, y + 1], C2[x + 0, y + 1], conj(iPEPS[x + 0, y + 1]), T2[x + 0, y + 1], iPEPS[x + 0, y + 1]);
    @ein rhoU[-1, -2, -3, -4, -5, -6] := rhoUL[-1, -2, -3, 1, 2, 3] * rhoUR[1, 2, 3, -4, -5, -6];

    rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[x + 1, y + 0], C4[x + 1, y + 0], conj(iPEPS[x + 1, y + 0]), T3[x + 1, y + 0], iPEPS[x + 1, y + 0]);
    rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[x + 1, y + 1], C3[x + 1, y + 1], conj(iPEPS[x + 1, y + 1]), T2[x + 1, y + 1], iPEPS[x + 1, y + 1]);
    @ein rhoD[-1, -2, -3, -4, -5, -6] := rhoDR[-1, -2, -3, 1, 2, 3] * rhoDL[1, 2, 3, -4, -5, -6];

    return rhoU, rhoD

end


#=
    initialize iPEPS
=#


Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];

chiB = 3;
chiE = 5;
truncBelowE = 1e-6;
d = 2;

latticeTens = Dict{Tuple{Int, Int}, Array{Float64, 5}}();
for idx = 1 : Lx, idy = 1 : Ly
    push!(latticeTens, (idx, idy) => randn(chiB, chiB+1, d, chiB, chiB+1));
end
iPEPS = pepsUnitCell(Lx, Ly, latticeTens, unitCellLayout);

# initialize CTMRG tensors
CTMRGTensors = initializeCTMRGTensors(iPEPS, chiE, initMethod = 0);
C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;

T1 = CTMRGTensors[2];
T3 = CTMRGTensors[6];
projsUL, projsDL = similar.((T1, T3));
projsUR, projsDR = similar.((T3, T1));
nC4, nT4, nC1 = similar.((C4, T4, C1));

foreach(keys(iPEPS.tensorDict)) do tensorKey
    projsUL[tensorKey...], projsDL[tensorKey...], projsUR[tensorKey...], projsDR[tensorKey...] = computeIsometries_LR(CTMRGTensors..., iPEPS, chiE, truncBelowE, tensorKey);
end

(oC1, oT1, oT3, oC4, oT4) = deepcopy.((C1, T1, T3, C4, T4));
foreach(keys(iPEPS.tensorDict)) do tensorKey
    @info tensorKey
    absorption_L!(C1, T1, T3, C4, T4, oC1, oT1, oT3, oC4, oT4, iPEPS, projsUL, projsDL, tensorKey);
end