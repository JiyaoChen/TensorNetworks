
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

    # normalize
    C1, T1, C2, T2, C3, T3, C4, T4 = normalizeCTMTensors(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS)
    # return (C1, T1, C2, T2, C3, T3, C4, T4, sinVals)
    return (C1, T1, C2, T2, C3, T3, C4, T4)

end

function normalizeCTMTensors(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS)
    nC1, nT1, nC2, nT2, nC3, nT3, nC4, nT4 = similar.((C1, T1, C2, T2, C3, T3, C4, T4));
    foreach(keys(iPEPS.tensorDict)) do tensorKey
        nC1[tensorKey...] = C1[tensorKey...]/norm(C1[tensorKey...])
        nC2[tensorKey...] = C2[tensorKey...]/norm(C2[tensorKey...])
        nC3[tensorKey...] = C3[tensorKey...]/norm(C3[tensorKey...])
        nC4[tensorKey...] = C4[tensorKey...]/norm(C4[tensorKey...])
        nT1[tensorKey...] = T1[tensorKey...]/norm(T1[tensorKey...])
        nT2[tensorKey...] = T2[tensorKey...]/norm(T2[tensorKey...])
        nT3[tensorKey...] = T3[tensorKey...]/norm(T3[tensorKey...])
        nT4[tensorKey...] = T4[tensorKey...]/norm(T4[tensorKey...])
    end
    return nC1, nT1, nC2, nT2, nC3, nT3, nC4, nT4
end