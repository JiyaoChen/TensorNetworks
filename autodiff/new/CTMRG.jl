include("absorptions.jl")
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

    # absorb uni-directional
    C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;
    C1, T1, C2 = absorptionStep_U(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    C4, T4, C1 = absorptionStep_L(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    # C3, T3, C4 = absorptionStep_D(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);
    # C2, T2, C3 = absorptionStep_R(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE);

    return (C1, T1, C2, T2, C3, T3, C4, T4)

end