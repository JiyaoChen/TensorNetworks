function truncateUSV(U, S, V, newChi)
    return U[:, 1 : newChi], S[1 : newChi], V[1 : newChi, :];
end

function computeProjectorsL_1(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)

    # compute upper and lower half of the network
    rhoU, rhoD = horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    sizeRhoU = size(rhoU);
    rhoU = reshape(rhoU, prod(sizeRhoU[1 : 3]), prod(sizeRhoU[4 : 6]));
    rhoU /= norm(rhoU);
    UU, SU, VU = svd(rhoU);
    VU = VU';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SU)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SU .> truncBelowE);
    # UU, SU, VU = truncateUSV(UU, SU, VU, newChi);

    # absorb sqrt(SU) into UU and VU
    sqrtSU = diagm(sqrt.(SU));
    FUL = UU * sqrtSU;
    # FUR = sqrtSU * VU;


    # perform SVD of rhoD
    sizeRhoD = size(rhoD);
    rhoD = reshape(rhoD, prod(sizeRhoD[1 : 3]), prod(sizeRhoD[4 : 6]));
    rhoD /= norm(rhoD);
    UD, SD, VD = svd(rhoD);
    VD = VD';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SD)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SD .> truncBelowE);
    # UD, SD, VD = truncateUSV(UD, SD, VD, newChi);

    # absorb sqrt(SD) into UD and VD
    sqrtSD = diagm(sqrt.(SD));
    # FDR = UD * sqrtSD;
    FDL = sqrtSD * VD;


    # compute biorthogonalization of FUL and FDL tensors and truncate UL, SL and VL
    BOL = FDL * FUL;
    BOL /= norm(BOL);
    UL, SL, VL = svd(BOL);
    VL = VL';

    # truncate UL, SL and VL
    newChi = min(chiE, length(SL));
    UL, SL, VL = truncateUSV(UL, SL, VL, newChi);
    sqrtSL = diagm(pinv.(sqrt.(SL)));

    # build projectors for left truncation
    PUL = FUL * VL' * sqrtSL;
    PDL = sqrtSL * UL' * FDL;

    # reshape PUL and PDL
    PUL = reshape(PUL, sizeRhoU[1], sizeRhoU[2], sizeRhoU[3], size(PUL, 2));
    PDL = reshape(PDL, size(PDL, 1), sizeRhoD[4], sizeRhoD[5], sizeRhoD[6]);

    # return projectors
    return PUL, PDL

end

function computeProjectorsU_1(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)
    
    # compute left and right half of the network
    rhoL, rhoR = verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # perform SVD of rhoL
    sizeRhoL = size(rhoL);
    rhoL = reshape(rhoL, prod(sizeRhoL[1 : 3]), prod(sizeRhoL[4 : 6]));
    rhoL /= norm(rhoL);        
    UL, SL, VL = svd(rhoL);
    VL = VL';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SL)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SL .> truncBelowE);
    # UL, SL, VL = truncateUSV(UL, SL, VL, newChi);

    # absorb sqrt(SL) into UL and VL
    sqrtSL = diagm(sqrt.(SL))
    # FDL = UL * sqrtSL;
    FUL = sqrtSL * VL;


    # perform SVD of rhoR
    sizeRhoR = size(rhoR);
    rhoR = reshape(rhoR, prod(sizeRhoR[1 : 3]), prod(sizeRhoR[4 : 6]));
    rhoR /= norm(rhoR);
    UR, SR, VR = svd(rhoR);
    VR = VR';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SR)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SR .> truncBelowE);
    # UR, SR, VR = truncateUSV(UR, SR, VR, newChi);

    # absorb sqrt(SR) into UR and VR
    sqrtSR = diagm(sqrt.(SR))
    FUR = UR * sqrtSR;
    # FDR = sqrtSR * VR;


    # compute biorthogonalization of FUL and FUR tensors
    BOU = FUL * FUR;
    BOU /= norm(BOU);
    UU, SU, VU = svd(BOU);
    VU = VU';

    # truncate UU, SU and VU
    newChi = min(chiE, length(SU));
    UU, SU, VU = truncateUSV(UU, SU, VU, newChi);
    sqrtSU = diagm(pinv.(sqrt.(SU)));

    # build projectors for up truncation
    PUL = sqrtSU * UU' * FUL;
    PUR = FUR * VU' * sqrtSU;
    
    # reshape PUL and PUR
    PUL = reshape(PUL, size(PUL, 1), sizeRhoL[4], sizeRhoL[5], sizeRhoL[6]);
    PUR = reshape(PUR, sizeRhoR[1], sizeRhoR[2], sizeRhoR[3], size(PUR, 2));
    
    # return projectors
    return  PUL, PUR;

end

function computeProjectorsR_1(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)

    # compute upper and lower half of the network
    rhoU, rhoD = horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    sizeRhoU = size(rhoU);
    rhoU = reshape(rhoU, prod(sizeRhoU[1 : 3]), prod(sizeRhoU[4 : 6]));
    rhoU /= norm(rhoU);
    UU, SU, VU = svd(rhoU);
    VU = VU';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SU)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SU .> truncBelowE);
    # UU, SU, VU = truncateUSV(UU, SU, VU, newChi);

    # absorb sqrt(SU) into UU and VU
    sqrtSU = diagm(sqrt.(SU));
    # FUL = UU * sqrtSU;
    FUR = sqrtSU * VU;


    # perform SVD of rhoD
    sizeRhoD = size(rhoD);
    rhoD = reshape(rhoD, prod(sizeRhoD[1 : 3]), prod(sizeRhoD[4 : 6]));
    rhoD /= norm(rhoD);
    UD, SD, VD = svd(rhoD);
    VD = VD';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SD)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SD .> truncBelowE);
    # UD, SD, VD = truncateUSV(UD, SD, VD, newChi);

    # absorb sqrt(SD) into UD and VD
    sqrtSD = diagm(sqrt.(SD));
    FDR = UD * sqrtSD;
    # FDL = sqrtSD * VD;


    # compute biorthogonalization of FUR and FDR tensors
    BOR = FUR * FDR;
    BOR /= norm(BOR);
    UR, SR, VR = svd(BOR);
    VR = VR';

    # truncate UR, SR and VR
    newChi = min(chiE, length(SR));
    UR, SR, VR = truncateUSV(UR, SR, VR, newChi);
    sqrtSR = diagm(pinv.(sqrt.(SR)));

    # build projectors for right truncation
    PUR = sqrtSR * UR' * FUR;
    PDR = FDR * VR' * sqrtSR;

    # reshape PUR and PDR
    PUR = reshape(PUR, size(PUR, 1), sizeRhoU[4], sizeRhoU[5], sizeRhoU[6]);
    PDR = reshape(PDR, sizeRhoD[1], sizeRhoD[2], sizeRhoD[3], size(PDR, 2));

    # return projectors
    return PUR, PDR;

end

function computeProjectorsD_1(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)
    
    # compute left and right half of the network
    rhoL, rhoR = verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # perform SVD of rhoL
    sizeRhoL = size(rhoL);
    rhoL = reshape(rhoL, prod(sizeRhoL[1 : 3]), prod(sizeRhoL[4 : 6]));
    rhoL /= norm(rhoL);        
    UL, SL, VL = svd(rhoL);
    VL = VL';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SL)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SL .> truncBelowE);
    # UL, SL, VL = truncateUSV(UL, SL, VL, newChi);

    # absorb sqrt(SL) into UL and VL
    sqrtSL = diagm(sqrt.(SL))
    FDL = UL * sqrtSL;
    # FUL = sqrtSL * VL;


    # perform SVD of rhoR
    sizeRhoR = size(rhoR);
    rhoR = reshape(rhoR, prod(sizeRhoR[1 : 3]), prod(sizeRhoR[4 : 6]));
    rhoR /= norm(rhoR);
    UR, SR, VR = svd(rhoR);
    VR = VR';

    # # truncate singular values
    # # caution, only truncate small singular values < ϵ (if at all), taking only min(chiE, length(SR)) singular values is too inaccurate to compute good projectors!
    # newChi = sum(SR .> truncBelowE);
    # UR, SR, VR = truncateUSV(UR, SR, VR, newChi);

    # absorb sqrt(SR) into UR and VR
    sqrtSR = diagm(sqrt.(SR))
    # FUR = UR * sqrtSR;
    FDR = sqrtSR * VR;


    # compute biorthogonalization of FDR and FDL tensors
    BOD = FDR * FDL;
    BOD /= norm(BOD);
    UD, SD, VD = svd(BOD);
    VD = VD';

    # truncate UD, SD and VD
    newChi = min(chiE, length(SD));
    UD, SD, VD = truncateUSV(UD, SD, VD, newChi);
    sqrtSD = diagm(pinv.(sqrt.(SD)));

    # build projectors for right truncation
    PDR = sqrtSD * UD' * FDR;
    PDL = FDL * VD' * sqrtSD;

    # reshape PDR and PDL
    PDR = reshape(PDR, size(PDR, 1), sizeRhoR[4], sizeRhoR[5], sizeRhoR[6]);
    PDL = reshape(PDL, sizeRhoL[1], sizeRhoL[2], sizeRhoL[3], size(PDL, 2));

    # return projectors
    return  PDL, PDR;

end



function computeProjectorsL_2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)

    # compute upper and lower half of the network
    rhoU, rhoD = horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # reshape rhoU to matrix
    sizeRhoU = size(rhoU);
    rhoU = reshape(rhoU, prod(sizeRhoU[1 : 3]), prod(sizeRhoU[4 : 6]));
    rhoU /= norm(rhoU);

    # reshape rhoD to matrix
    sizeRhoD = size(rhoD);
    rhoD = reshape(rhoD, prod(sizeRhoD[1 : 3]), prod(sizeRhoD[4 : 6]));
    rhoD /= norm(rhoD);

    # compute biorthogonalization of rhoD and rhoU
    BOL = rhoD * rhoU;
    BOL /= norm(BOL);
    UL, SL, VL = svd(BOL);
    VL = VL';

    # truncate UL, SL and VL
    newChi = min(chiE, length(SL));
    UL, SL, VL = truncateUSV(UL, SL, VL, newChi);
    sqrtSL = diagm(pinv.(sqrt.(SL)));

    # build projectors for left truncation
    PUL = rhoU * VL' * sqrtSL;
    PDL = sqrtSL * UL' * rhoD;

    # reshape PUL and PDL
    PUL = reshape(PUL, sizeRhoU[1], sizeRhoU[2], sizeRhoU[3], size(PUL, 2));
    PDL = reshape(PDL, size(PDL, 1), sizeRhoD[4], sizeRhoD[5], sizeRhoD[6]);

    # return projectors
    return PUL, PDL

end

function computeProjectorsU_2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)
    
    # compute left and right half of the network
    rhoL, rhoR = verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # reshape rhoL to matrix
    sizeRhoL = size(rhoL);
    rhoL = reshape(rhoL, prod(sizeRhoL[1 : 3]), prod(sizeRhoL[4 : 6]));
    rhoL /= norm(rhoL);

    # reshape rhoR to matrix
    sizeRhoR = size(rhoR);
    rhoR = reshape(rhoR, prod(sizeRhoR[1 : 3]), prod(sizeRhoR[4 : 6]));
    rhoR /= norm(rhoR);

    # compute biorthogonalization of rhoL and rhoR
    BOU = rhoL * rhoR;
    BOU /= norm(BOU);
    UU, SU, VU = svd(BOU);
    VU = VU';

    # truncate UU, SU and VU
    newChi = min(chiE, length(SU));
    UU, SU, VU = truncateUSV(UU, SU, VU, newChi);
    sqrtSU = diagm(pinv.(sqrt.(SU)));

    # build projectors for up truncation
    PUL = sqrtSU * UU' * rhoL;
    PUR = rhoR * VU' * sqrtSU;
    
    # reshape PUL and PUR
    PUL = reshape(PUL, size(PUL, 1), sizeRhoL[4], sizeRhoL[5], sizeRhoL[6]);
    PUR = reshape(PUR, sizeRhoR[1], sizeRhoR[2], sizeRhoR[3], size(PUR, 2));
    
    # return projectors
    return  PUL, PUR;

end

function computeProjectorsR_2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)

    # compute upper and lower half of the network
    rhoU, rhoD = horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # reshape rhoU to matrix
    sizeRhoU = size(rhoU);
    rhoU = reshape(rhoU, prod(sizeRhoU[1 : 3]), prod(sizeRhoU[4 : 6]));
    rhoU /= norm(rhoU);

    # reshape rhoD to matrix
    sizeRhoD = size(rhoD);
    rhoD = reshape(rhoD, prod(sizeRhoD[1 : 3]), prod(sizeRhoD[4 : 6]));
    rhoD /= norm(rhoD);

    # compute biorthogonalization of rhoU and rhoD
    BOR = rhoU * rhoD;
    BOR /= norm(BOR);
    UR, SR, VR = svd(BOR);
    VR = VR';

    # truncate UR, SR and VR
    newChi = min(chiE, length(SR));
    UR, SR, VR = truncateUSV(UR, SR, VR, newChi);
    sqrtSR = diagm(pinv.(sqrt.(SR)));

    # build projectors for right truncation
    PUR = sqrtSR * UR' * rhoU;
    PDR = rhoD * VR' * sqrtSR;

    # reshape PUR and PDR
    PUR = reshape(PUR, size(PUR, 1), sizeRhoU[4], sizeRhoU[5], sizeRhoU[6]);
    PDR = reshape(PDR, sizeRhoD[1], sizeRhoD[2], sizeRhoD[3], size(PDR, 2));

    # return projectors
    return PUR, PDR;

end

function computeProjectorsD_2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (idX, idY), absDir)
    
    # compute left and right half of the network
    rhoL, rhoR = verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (idX, idY), absDir)

    # reshape rhoL to matrix
    sizeRhoL = size(rhoL);
    rhoL = reshape(rhoL, prod(sizeRhoL[1 : 3]), prod(sizeRhoL[4 : 6]));
    rhoL /= norm(rhoL);

    # reshape rhoR to matrix
    sizeRhoR = size(rhoR);
    rhoR = reshape(rhoR, prod(sizeRhoR[1 : 3]), prod(sizeRhoR[4 : 6]));
    rhoR /= norm(rhoR);

    # compute biorthogonalization of rhoR and rhoL tensors
    BOD = rhoR * rhoL;
    BOD /= norm(BOD);
    UD, SD, VD = svd(BOD);
    VD = VD';

    # truncate UD, SD and VD
    newChi = min(chiE, length(SD));
    UD, SD, VD = truncateUSV(UD, SD, VD, newChi);
    sqrtSD = diagm(pinv.(sqrt.(SD)));

    # build projectors for right truncation
    PDR = sqrtSD * UD' * rhoR;
    PDL = rhoL * VD' * sqrtSD;

    # reshape PDR and PDL
    PDR = reshape(PDR, size(PDR, 1), sizeRhoR[4], sizeRhoR[5], sizeRhoR[6]);
    PDL = reshape(PDL, sizeRhoL[1], sizeRhoL[2], sizeRhoL[3], size(PDL, 2));

    # return projectors
    return  PDL, PDR;

end



function horizontalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (x, y), absDir)

    Lx, Ly = size(pepsTensors);

    # contract the four corners of the TN
    if absDir == 'L'
        rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...]);
        rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], C4[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...]), T3[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], C3[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...]);
    elseif absDir == 'R'
        rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...]), T1[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...]);
        rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[getCoordinates(x + 0, Lx, y - 0, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y - 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y - 0, Ly, unitCellLayout)...]), T2[getCoordinates(x + 0, Lx, y - 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y - 0, Ly, unitCellLayout)...]);
        rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[getCoordinates(x + 1, Lx, y - 1, Ly, unitCellLayout)...], C4[getCoordinates(x + 1, Lx, y - 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y - 1, Ly, unitCellLayout)...]), T3[getCoordinates(x + 1, Lx, y - 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y - 1, Ly, unitCellLayout)...]);
        rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[getCoordinates(x + 1, Lx, y - 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 1, Lx, y - 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y - 0, Ly, unitCellLayout)...]), T2[getCoordinates(x + 1, Lx, y - 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y - 0, Ly, unitCellLayout)...]);
    end
    
    # contract upper and lower half
    @ein rhoU[-1, -2, -3, -4, -5, -6] := rhoUL[-1, -2, -3, 1, 2, 3] * rhoUR[1, 2, 3, -4, -5, -6];
    @ein rhoD[-1, -2, -3, -4, -5, -6] := rhoDR[-1, -2, -3, 1, 2, 3] * rhoDL[1, 2, 3, -4, -5, -6];

    return rhoU, rhoD;

end

function verticalCut(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, (x, y), absDir)

    Lx, Ly = size(pepsTensors);

    # contract the four corners of the TN
    if absDir == 'U'
        rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 1, Ly, unitCellLayout)...]);
        rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], C4[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...]), T3[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], C3[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 1, Lx, y + 1, Ly, unitCellLayout)...]);
    elseif absDir == 'D'
        rhoUL = ein"(((ahkg, gi), hcmfj), iljd), kbmel -> abcdef"(T4[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...]), T1[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoUR = ein"(((alji, ig), cemhj), khdg), bfmkl -> abcdef"(T1[getCoordinates(x - 1, Lx, y + 1, Ly, unitCellLayout)...], C2[getCoordinates(x - 1, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x - 1, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x - 1, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x - 1, Lx, y + 1, Ly, unitCellLayout)...]);
        rhoDL = ein"(((ghkd, ig), hjmbf), iajl), klmce -> abcdef"(T4[getCoordinates(x - 0, Lx, y + 0, Ly, unitCellLayout)...], C4[getCoordinates(x - 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x - 0, Lx, y + 0, Ly, unitCellLayout)...]), T3[getCoordinates(x - 0, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x - 0, Lx, y + 0, Ly, unitCellLayout)...]);
        rhoDR = ein"(((dijl, ig), ejmhb), khga), flmkc -> abcdef"(T3[getCoordinates(x - 0, Lx, y + 1, Ly, unitCellLayout)...], C3[getCoordinates(x - 0, Lx, y + 1, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x - 0, Lx, y + 1, Ly, unitCellLayout)...]), T2[getCoordinates(x - 0, Lx, y + 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x - 0, Lx, y + 1, Ly, unitCellLayout)...]);
    end

    # contract left and right half
    @ein rhoL[-1, -2, -3, -4, -5, -6] := rhoDL[-1, -2, -3, 1, 2, 3] * rhoUL[1, 2, 3, -4, -5, -6];
    @ein rhoR[-1, -2, -3, -4, -5, -6] := rhoUR[-1, -2, -3, 1, 2, 3] * rhoDR[1, 2, 3, -4, -5, -6];

    return rhoL, rhoR;

end