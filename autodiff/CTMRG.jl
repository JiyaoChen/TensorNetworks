
function runCTMRG(iPEPS, chiE, truncBelowE, convTol, maxIter, initMethod = 0)

    # initialize structs for CTMRG tensors
    CTMRGTensors = initializeCTMRGTensors(iPEPS, chiE, initMethod = initMethod);

    # # get physical dimension
    # d = size(iPEPS[1,1], 3);

    # initialize variables to control CTMRG procedure
    Lx = iPEPS.Lx;
    Ly = iPEPS.Ly;
    # sinVals = fill(Inf, Lx, Ly, 4 * chiE);
    sinVals = fill(Inf, Lx, Ly, 4, chiE);

    # construct struct for stopFunction
    stopFunc = StopFunction(sinVals, 0, convTol, maxIter, chiE);

    # make CTMRG step and return CRMTG tensors
    # CTMRGTensors = CTMRGStep((CTMRGTensors..., sinVals), (iPEPS, chiE, truncBelowE, d));

    # run fixedPoint CTMRG routine and return CTMRGTensors
    # CTMRGTensors = fixedPoint(CTMRGStep, (CTMRGTensors..., sinVals), (iPEPS, chiE, truncBelowE), stopFunc)[1 : end - 1];
    CTMRGTensors = fixedPoint(CTMRGStep, CTMRGTensors, (iPEPS, chiE, truncBelowE), stopFunc);
    return CTMRGTensors

end

# function CTMRGStep((C1, T1, C2, T2, C3, T3, C4, T4, sinVals), (iPEPS, chiE, truncBelowE))
function CTMRGStep(CTMRGTensors, (iPEPS, chiE, truncBelowE))

    # do bi-directinal absorptions towards LR and UD
    C1, T1, C2, T2, C3, T3, C4, T4 = CTMRGTensors;
    C4, T4, C1, C2, T2, C3 = absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE);
    C1, T1, C2, C3, T3, C4 = absorptionStep_UD(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE);

    return (C1, T1, C2, T2, C3, T3, C4, T4)

end