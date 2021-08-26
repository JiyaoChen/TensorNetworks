function energy(pepsTensors, unitCellLayout, CTMRGTensors, tbg)
    
    # get size
    Lx, Ly = size(pepsTensors);
    expValsH = [twoSiteExpVal_H(pepsTensors, unitCellLayout, CTMRGTensors, tbg, (idx, idy)) for idx = 1 : Lx, idy = 1 : Ly];
    expValsV = [twoSiteExpVal_V(pepsTensors, unitCellLayout, CTMRGTensors, tbg, (idx, idy)) for idx = 1 : Lx, idy = 1 : Ly];
    energy = 1 / (2 * Lx * Ly) * (sum(expValsH) + sum(expValsV));
    return energy
    
end

# function twoSiteExpVal_H(pepsTensorL, pepsTensorR, twoSiteOperator, envTensorsL, envTensorsR)
function twoSiteExpVal_H(pepsTensors, unitCellLayout, CTMRGTensors, twoSiteOperator, (idx, idy))

    # get size
    Lx, Ly = size(pepsTensors);

    pepsTensorL = pepsTensors[getCoordinates(idx + 0, Lx, idy + 0, Ly, unitCellLayout)...];
    pepsTensorR = pepsTensors[getCoordinates(idx + 0, Lx, idy + 1, Ly, unitCellLayout)...];
    
    envTensorsL = [CT[getCoordinates(idx + 0, Lx, idy + 0, Ly, unitCellLayout)...] for CT in CTMRGTensors];
    envTensorsR = [CT[getCoordinates(idx + 0, Lx, idy + 1, Ly, unitCellLayout)...] for CT in CTMRGTensors];

    TNL = ein"ae, ebcf, fd -> abcd"(envTensorsL[7], envTensorsL[8], envTensorsL[1]);
    TNL = ein"ahfg, hbecd -> abcdefg"(TNL, conj(pepsTensorL));
    TNL = ein"hibcdeg, haif -> abcdefg"(TNL, envTensorsL[6]);
    TNL = ein"aeidbch, hfig -> abcdefg"(TNL, envTensorsL[2]);
    TNL = ein"dghfcia, ghebi -> abcdef"(TNL, pepsTensorL);

    TNR = ein"ae, bcfe, df -> abcd"(envTensorsR[3], envTensorsR[4], envTensorsR[5]);
    TNR = ein"abhg, efdhc -> abcdefg"(TNR, conj(pepsTensorR));
    TNR = ein"abcdeih, ghif -> abcdefg"(TNR, envTensorsR[6]);
    TNR = ein"hficdeg, abih -> abcdefg"(TNR, envTensorsR[2]);
    TNR = ein"agfcihd, biehg -> abcdef"(TNR, pepsTensorR);

    # compute norm
    expVal_N = ein"abcdee, abcdff -> "(TNL, TNR)[];

    # compute expectation value
    TNGateL = ein"abcdgh, gehf -> abcdef"(TNL, twoSiteOperator);
    expVal_O = ein"abcdef, abcdef -> "(TNGateL, TNR)[];
    # println(expVal_O)

    # sizeTNL = size(TNGateL);
    # sizeTNR = size(TNR);
    # TNGateL = reshape(TNGateL, prod(sizeTNL));
    # TNR = reshape(TNR, prod(sizeTNR));
    # expVal_O = dot(TNGateL, TNR);
    # println(expVal_O)
    
    # println("Energy / Norm: $expVal_O / $expVal_N")
    expVal = real(expVal_O) / real(expVal_N);
    return expVal;

end

function twoSiteExpVal_V(pepsTensors, unitCellLayout, CTMRGTensors, twoSiteOperator, (idx, idy))

    # get size
    Lx, Ly = size(pepsTensors);

    pepsTensorU = pepsTensors[getCoordinates(idx + 0, Lx, idy + 0, Ly, unitCellLayout)...];
    pepsTensorD = pepsTensors[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...];
    
    envTensorsU = [CT[getCoordinates(idx + 0, Lx, idy + 0, Ly, unitCellLayout)...] for CT in CTMRGTensors];
    envTensorsD = [CT[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...] for CT in CTMRGTensors];

    TNU = ein"(((((dhng, gi), hbfmj), ipjk), kl), omal), nceop -> abcdef"(envTensorsU[8], envTensorsU[1], conj(pepsTensorU), envTensorsU[2], envTensorsU[3], envTensorsU[4], pepsTensorU);
    TND = ein"(((((ghnd, ig), hjfmb), ikjo), kl), pmla), noepc -> abcdef"(envTensorsD[8], envTensorsD[7], conj(pepsTensorD), envTensorsD[6], envTensorsD[5], envTensorsD[4], pepsTensorD);

    # compute norm
    expVal_N = ein"abcdee, abcdff -> "(TND, TNU)[];

    # compute expectation value
    TNGateD = ein"abcdgh, gehf -> abcdef"(TND, twoSiteOperator);
    expVal_O = ein"abcdef, abcdef -> "(TNGateD, TNU)[];
    # println(expVal_O)

    # sizeTNL = size(TNGateL);
    # sizeTNR = size(TNR);
    # TNGateL = reshape(TNGateL, prod(sizeTNL));
    # TNR = reshape(TNR, prod(sizeTNR));
    # expVal_O = dot(TNGateL, TNR);
    # println(expVal_O)
    
    # println("Energy / Norm: $expVal_O / $expVal_N")
    expVal = real(expVal_O) / real(expVal_N);
    return expVal;

end