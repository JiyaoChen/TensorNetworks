
include("absorptions.jl")
include("absorptions2.jl")
include("computeIsometries.jl")
include("fixedpoint.jl")

function runCTMRG(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod = 0)

    # get size
    Lx, Ly = size(pepsTensors);

    # get element type and dimensions of pepsTensors
    elementType = eltype(eltype(pepsTensors));
    dimensionsTensors = [size(pepsTensors[idx, idy]) for idx = 1 : Lx, idy = 1 : Ly];

    # initialize structs for CTMRG tensors
    CTMRGTensors = initializeCTMRGTensors(elementType, dimensionsTensors, unitCellLayout, chiE, initMethod = initMethod);

    # sinVals = fill(Inf, Lx, Ly, 4 * chiE);
    sinVals = fill(fill(Inf, 4 * chiE), Lx, Ly);

    # construct struct for stopFunction
    stopFunc = StopFunction(Lx, Ly, sinVals, 0, convTol, maxIter, chiE);

    # make CTMRG step and return CRMTG tensors
    # CTMRGTensors = CTMRGStep((CTMRGTensors..., sinVals), (pepsTensors, chiE, truncBelowE, d));

    # run fixedPoint CTMRG routine and return CTMRGTensors
    CTMRGTensors = fixedPoint(CTMRGStep, CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE), stopFunc);
    # CTMRGTensors = fixedPointAD(CTMRGStep, CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE), stopFunc);
    return CTMRGTensors

end

# function CTMRGStep((C1, T1, C2, T2, C3, T3, C4, T4, sinVals), (pepsTensors, chiE, truncBelowE))
function CTMRGStep(CTMRGTensors, (pepsTensors, unitCellLayout, chiE, truncBelowE))

    Lx, Ly = size(pepsTensors);

    # environmentTensors = Array{Any, 2}(undef, Lx, Ly);

    # absorb uni-directional
    C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;
    C4, T4, C1 = absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    C1, T1, C2 = absorptionStep_U2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    C2, T2, C3 = absorptionStep_R2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    C3, T3, C4 = absorptionStep_D2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    # # absorb uni-directional
    # C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;

    # C4, T4, C1 = absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     environmentTensors[idx, idy] = [C1[idx, idy], T1[idx, idy], C2[idx, idy], T2[idx, idy], C3[idx, idy], T3[idx, idy], C4[idx, idy], T4[idx, idy]];
    # end

    # pepsTensors, unitCellLayout, environmentTensors = rotateLatticePEPS_A90(pepsTensors, unitCellLayout, environmentTensors);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     C1[idx, idy] = environmentTensors[idx, idy][1];
    #     T1[idx, idy] = environmentTensors[idx, idy][2];
    #     C2[idx, idy] = environmentTensors[idx, idy][3];
    #     T2[idx, idy] = environmentTensors[idx, idy][4];
    #     C3[idx, idy] = environmentTensors[idx, idy][5];
    #     T3[idx, idy] = environmentTensors[idx, idy][6];
    #     C4[idx, idy] = environmentTensors[idx, idy][7];
    #     T4[idx, idy] = environmentTensors[idx, idy][8];
    # end

    # C4, T4, C1 = absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     environmentTensors[idx, idy] = [C1[idx, idy], T1[idx, idy], C2[idx, idy], T2[idx, idy], C3[idx, idy], T3[idx, idy], C4[idx, idy], T4[idx, idy]];
    # end

    # pepsTensors, unitCellLayout, environmentTensors = rotateLatticePEPS_A90(pepsTensors, unitCellLayout, environmentTensors);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     C1[idx, idy] = environmentTensors[idx, idy][1];
    #     T1[idx, idy] = environmentTensors[idx, idy][2];
    #     C2[idx, idy] = environmentTensors[idx, idy][3];
    #     T2[idx, idy] = environmentTensors[idx, idy][4];
    #     C3[idx, idy] = environmentTensors[idx, idy][5];
    #     T3[idx, idy] = environmentTensors[idx, idy][6];
    #     C4[idx, idy] = environmentTensors[idx, idy][7];
    #     T4[idx, idy] = environmentTensors[idx, idy][8];
    # end

    # C4, T4, C1 = absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     environmentTensors[idx, idy] = [C1[idx, idy], T1[idx, idy], C2[idx, idy], T2[idx, idy], C3[idx, idy], T3[idx, idy], C4[idx, idy], T4[idx, idy]];
    # end

    # pepsTensors, unitCellLayout, environmentTensors = rotateLatticePEPS_A90(pepsTensors, unitCellLayout, environmentTensors);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     C1[idx, idy] = environmentTensors[idx, idy][1];
    #     T1[idx, idy] = environmentTensors[idx, idy][2];
    #     C2[idx, idy] = environmentTensors[idx, idy][3];
    #     T2[idx, idy] = environmentTensors[idx, idy][4];
    #     C3[idx, idy] = environmentTensors[idx, idy][5];
    #     T3[idx, idy] = environmentTensors[idx, idy][6];
    #     C4[idx, idy] = environmentTensors[idx, idy][7];
    #     T4[idx, idy] = environmentTensors[idx, idy][8];
    # end

    # C4, T4, C1 = absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     environmentTensors[idx, idy] = [C1[idx, idy], T1[idx, idy], C2[idx, idy], T2[idx, idy], C3[idx, idy], T3[idx, idy], C4[idx, idy], T4[idx, idy]];
    # end

    # pepsTensors, unitCellLayout, environmentTensors = rotateLatticePEPS_A90(pepsTensors, unitCellLayout, environmentTensors);

    # for idx = 1 : Lx, idy = 1 : Ly
    #     C1[idx, idy] = environmentTensors[idx, idy][1];
    #     T1[idx, idy] = environmentTensors[idx, idy][2];
    #     C2[idx, idy] = environmentTensors[idx, idy][3];
    #     T2[idx, idy] = environmentTensors[idx, idy][4];
    #     C3[idx, idy] = environmentTensors[idx, idy][5];
    #     T3[idx, idy] = environmentTensors[idx, idy][6];
    #     C4[idx, idy] = environmentTensors[idx, idy][7];
    #     T4[idx, idy] = environmentTensors[idx, idy][8];
    # end

    

    # # absorb bi-directional
    # C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;
    # C4, T4, C1, C2, T2, C3 = absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    # C1, T1, C2, C3, T3, C4 = absorptionStep_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    return (C1, T1, C2, T2, C3, T3, C4, T4)

end

function rotateEnvTensors(envTensors)
    return [permutedims(envTensors[3], [1, 2]), permutedims(envTensors[4], [4, 1, 2, 3]), permutedims(envTensors[5], [2, 1]), permutedims(envTensors[6], [4, 3, 1, 2]), permutedims(envTensors[7], [2, 1]), permutedims(envTensors[8], [4, 1, 2, 3]), permutedims(envTensors[1], [1, 2]), permutedims(envTensors[2], [1, 3, 2, 4])];
end


function rotateLatticePEPS_A90(pepsTensors, unitCell, environmentTensors)

    # get unit cell size
    Lx, Ly = size(pepsTensors);

    # rotate unit cell of tensors
    pepsTensors = [permutedims(pepsTensors[idx,idy], [5, 1, 3, 2, 4]) for idx = 1 : Lx, idy = 1 : Ly];

    # rearrange the full unitCell
    unitCell = rotl90(unitCell);

    # permute tensor indices for envTensors
    environmentTensors = rotateEnvTensors.(environmentTensors);

    # function return
    return pepsTensors, unitCell, environmentTensors;

end