
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
    stopFunc = StopFunction(sinVals, 0, convTol, maxIter);

    # make CTMRG step and return CRMTG tensors
    # CTMRGTensors = CTMRGStep((CTMRGTensors..., sinVals), (iPEPS, chiE, truncBelowE, d));

    # run fixedPoint CTMRG routine and return CTMRGTensors
    CTMRGTensors = fixedPoint(CTMRGStep, (CTMRGTensors..., sinVals), (iPEPS, chiE, truncBelowE), stopFunc)[1 : end - 1];
    return CTMRGTensors

end

function CTMRGStep((C1, T1, C2, T2, C3, T3, C4, T4, sinVals), (iPEPS, chiE, truncBelowE))

    C4, T4, C1, C2, T2, C3 = absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE);
    C1, T1, C2, C3, T3, C4 = absorptionStep_UD(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE);


    # normalize
    normalizeCTMTensors!(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS)
    
    return (C1, T1, C2, T2, C3, T3, C4, T4, sinVals)

end

function normalizeCTMTensors!(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS)
    foreach(keys(iPEPS.tensorDict)) do tensorKey
        C1[tensorKey...] /= norm(C1[tensorKey...])
        C2[tensorKey...] /= norm(C2[tensorKey...])
        C3[tensorKey...] /= norm(C3[tensorKey...])
        C4[tensorKey...] /= norm(C4[tensorKey...])
        T1[tensorKey...] /= norm(T1[tensorKey...])
        T2[tensorKey...] /= norm(T2[tensorKey...])
        T3[tensorKey...] /= norm(T3[tensorKey...])
        T4[tensorKey...] /= norm(T4[tensorKey...])
    end
end