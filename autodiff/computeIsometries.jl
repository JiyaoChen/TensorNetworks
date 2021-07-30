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



    # compute biorthogonalization of FUL and FDL tensors and truncate UL, SL and VL
    BOL = FDL * FUL;
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
    PUL = FUL * VL' * sqrtSL;
    PDL = sqrtSL * UL' * FDL;

    # reshape PUL and PDL
    PUL = reshape(PUL, sizeRhoU[1], sizeRhoU[2], sizeRhoU[3], size(PUL, 2));
    PDL = reshape(PDL, size(PDL, 1), sizeRhoD[4], sizeRhoD[5], sizeRhoD[6]);


    # compute biorthogonalization of FUR and FDR tensors
    BOR = FUR * FDR;
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
    PUR = sqrtSR * UR' * FUR;
    PDR = FDR * VR' * sqrtSR;

    # reshape PUR and PDR
    PUR = reshape(PUR, size(PUR, 1), sizeRhoU[4], sizeRhoU[5], sizeRhoU[6]);
    PDR = reshape(PDR, sizeRhoD[1], sizeRhoD[2], sizeRhoD[3], size(PDR, 2));


    # return projectors
    return  PUL, PDL, PUR, PDR;

end

function computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE, (idX, idY))
    
    # compute left and right half of the network
    rhoL, rhoR = verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, (idX, idY))
    d = size(iPEPS[1,1], 3);

    # perform SVD of rhoL
    sizeRhoL = size(rhoL);
    rhoL = reshape(rhoL, prod(sizeRhoL[1 : 3]), prod(sizeRhoL[4 : 6]));
    rhoL /= norm(rhoL);        
    UL, SL, VL = svd(rhoL);
    SL = diagm(SL);
    VL = VL';

    # truncate singular values
    newChi = sum(diag(SL) .> truncBelowE);
    UL = UL[:, 1 : newChi];
    SL = SL[1 : newChi, 1 : newChi];
    VL = VL[1 : newChi, :];

    # absorb sqrt(SL) into UL and VL
    FDL = UL * sqrt(SL);
    FUL = sqrt(SL) * VL;


    # perform SVD of rhoR
    sizeRhoR = size(rhoR);
    rhoR = reshape(rhoR, prod(sizeRhoR[1 : 3]), prod(sizeRhoR[4 : 6]));
    rhoR /= norm(rhoR);
    UR, SR, VR = svd(rhoR);
    SR = diagm(SR);
    VR = VR';

    # truncate singular values
    newChi = sum(diag(SR) .> truncBelowE);
    UR = UR[:, 1 : newChi];
    SR = SR[1 : newChi, 1 : newChi];
    VR = VR[1 : newChi, :];

    # absorb sqrt(SR) into UR and VR
    FUR = UR * sqrt(SR);
    FDR = sqrt(SR) * VR;


    # compute biorthogonalization of FUL and FUR tensors
    BOU = FUL * FUR;
    BOU /= norm(BOU);
    UU, SU, VU = svd(BOU);
    SU = diagm(SU);
    VU = VU';

    # truncate UU, SU and VU
    newChi = min(chiE, sum(diag(SU) .> truncBelowE));
    UU = UU[:, 1 : newChi];
    SU = SU[1 : newChi, 1 : newChi];
    VU = VU[1 : newChi, :];
    sqrtSU = sqrt(pinv(SU));

    # build projectors for up truncation
    PUL = sqrtSU * UU' * FUL;
    PUR = FUR * VU' * sqrtSU;
    
    # reshape PUL and PUR
    PUL = reshape(PUL, size(PUL, 1), sizeRhoL[4], sizeRhoL[5], sizeRhoL[6]);
    PUR = reshape(PUR, sizeRhoR[1], sizeRhoR[2], sizeRhoR[3], size(PUR, 2));
    

    # compute biorthogonalization of FDR and FDL tensors
    BOD = FDR * FDL;
    BOD /= norm(BOD);
    UD, SD, VD = svd(BOD);
    SD = diagm(SD);
    VD = VD';

    # truncate UD, SD and VD
    newChi = min(chiE, sum(diag(SD) .> truncBelowE));
    UD = UD[:, 1 : newChi];
    SD = SD[1 : newChi, 1 : newChi];
    VD = VD[1 : newChi, :];
    sqrtSD = sqrt(pinv(SD));

    # build projectors for right truncation
    PDR = sqrtSD * UD' * FDR;
    PDL = FDL * VD' * sqrtSD;

    # reshape PDR and PDL
    PDR = reshape(PDR, size(PDR, 1), sizeRhoR[4], sizeRhoR[5], sizeRhoR[6]);
    PDL = reshape(PDL, sizeRhoL[1], sizeRhoL[2], sizeRhoL[3], size(PDL, 2));


    # return projectors
    return  PUL, PUR, PDL, PDR;

end

function horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, (x, y))

    # contract the four corners of the TN
    rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[x + 0, y + 0], C1[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), T1[x + 0, y + 0], iPEPS[x + 0, y + 0]);
    rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[x + 0, y + 1], C2[x + 0, y + 1], conj(iPEPS[x + 0, y + 1]), T2[x + 0, y + 1], iPEPS[x + 0, y + 1]);
    rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[x + 1, y + 0], C4[x + 1, y + 0], conj(iPEPS[x + 1, y + 0]), T3[x + 1, y + 0], iPEPS[x + 1, y + 0]);
    rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[x + 1, y + 1], C3[x + 1, y + 1], conj(iPEPS[x + 1, y + 1]), T2[x + 1, y + 1], iPEPS[x + 1, y + 1]);
    
    # contract upper and lower half
    @ein rhoU[-1, -2, -3, -4, -5, -6] := rhoUL[-1, -2, -3, 1, 2, 3] * rhoUR[1, 2, 3, -4, -5, -6];
    @ein rhoD[-1, -2, -3, -4, -5, -6] := rhoDR[-1, -2, -3, 1, 2, 3] * rhoDL[1, 2, 3, -4, -5, -6];

    return rhoU, rhoD

end

function verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, (x, y))

    # contract the four corners of the TN
    rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[x + 0, y + 0], C1[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), T1[x + 0, y + 0], iPEPS[x + 0, y + 0]);
    rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[x + 0, y + 1], C2[x + 0, y + 1], conj(iPEPS[x + 0, y + 1]), T2[x + 0, y + 1], iPEPS[x + 0, y + 1]);
    rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[x + 1, y + 0], C4[x + 1, y + 0], conj(iPEPS[x + 1, y + 0]), T3[x + 1, y + 0], iPEPS[x + 1, y + 0]);
    rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[x + 1, y + 1], C3[x + 1, y + 1], conj(iPEPS[x + 1, y + 1]), T2[x + 1, y + 1], iPEPS[x + 1, y + 1]);

    # contract left and right half
    @ein rhoL[-1, -2, -3, -4, -5, -6] := rhoDL[-1, -2, -3, 1, 2, 3] * rhoUL[1, 2, 3, -4, -5, -6];
    @ein rhoR[-1, -2, -3, -4, -5, -6] := rhoUR[-1, -2, -3, 1, 2, 3] * rhoDR[1, 2, 3, -4, -5, -6];

    return rhoL, rhoR

end