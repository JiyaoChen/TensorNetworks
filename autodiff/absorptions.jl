function absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # projsUL, projsDL = similar.((T1, T3));
    # projsUR, projsDR = similar.((T3, T1));
    # nC4, nT4, nC1 = similar.((C4, T4, C1));
    # nC2, nT2, nC3 = similar.((C2, T2, C3));

    # foreach(keys(pepsTensors.tensorDict)) do tensorKey
    #     projsUL[tensorKey...], projsDL[tensorKey...], projsUR[tensorKey...], projsDR[tensorKey...] = computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, chiE, truncBelowE, tensorKey);
    # end

    # (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    # foreach(keys(pepsTensors.tensorDict)) do tensorKey
    #     absorption_L!(nC4, nT4, nC1, oC1, oT1, oT3, oC4, oT4, pepsTensors, projsUL, projsDL, tensorKey);
    #     absorption_R!(nC2, nT2, nC3, oT1, oC2, oT2, oC3, oT3, pepsTensors, projsUR, projsDR, tensorKey);
    # end

    # get size
    Lx, Ly = size(pepsTensors);
    allProjectors = [computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    projsUL = [proj[1] for proj in allProjectors];
    projsDL = [proj[2] for proj in allProjectors];
    projsUR = [proj[3] for proj in allProjectors];
    projsDR = [proj[4] for proj in allProjectors];
    # projsUL = pepsUnitCell(Lx, Ly, projsUL, unitCellLayout);
    # projsDL = pepsUnitCell(Lx, Ly, projsDL, unitCellLayout);
    # projsUR = pepsUnitCell(Lx, Ly, projsUR, unitCellLayout);
    # projsDR = pepsUnitCell(Lx, Ly, projsDR, unitCellLayout);

    # (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    allTensorsL = [absorption_L(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    nC4 = [allTensorsL[getCoordinates(idx + 0, Lx, idy + 1, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
    nT4 = [allTensorsL[getCoordinates(idx + 0, Lx, idy + 1, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
    nC1 = [allTensorsL[getCoordinates(idx + 0, Lx, idy + 1, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
    # nC4 = pepsUnitCell(Lx, Ly, nC4, unitCellLayout);
    # nT4 = pepsUnitCell(Lx, Ly, nT4, unitCellLayout);
    # nC1 = pepsUnitCell(Lx, Ly, nC1, unitCellLayout);

    allTensorsR = [absorption_R(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    nC2 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
    nT2 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
    nC3 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
    # nC2 = pepsUnitCell(Lx, Ly, nC2, unitCellLayout);
    # nT2 = pepsUnitCell(Lx, Ly, nT2, unitCellLayout);
    # nC3 = pepsUnitCell(Lx, Ly, nC3, unitCellLayout);

    return nC4, nT4, nC1, nC2, nT2, nC3

end

function absorptionStep_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)
    
    # projsUL, projsUR = similar.((T3, T1));
    # projsDL, projsDR = similar.((T1, T3));
    # nC1, nT1, nC2 = similar.((C1, T1, C2));
    # nC3, nT3, nC4 = similar.((C3, T3, C4));

    # tensorKeys = keys(pepsTensors.tensorDict)
    # # foreach(keys(pepsTensors.tensorDict)) do tensorKey
    # foreach(tensorKeys) do tensorKey
    #     projsUL[tensorKey...], projsUR[tensorKey...], projsDL[tensorKey...], projsDR[tensorKey...] = computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, chiE, truncBelowE, tensorKey);
    # end

    # (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    # foreach(keys(pepsTensors.tensorDict)) do tensorKey
    #     absorption_U!(nC1, nT1, nC2, oT4, oC1, oT1, oC2, oT2, pepsTensors, projsUL, projsUR, tensorKey);
    #     absorption_D!(nC3, nT3, nC4, oT2, oC3, oT3, oC4, oT4, pepsTensors, projsDL, projsDR, tensorKey);
    # end

    # get size
    Lx, Ly = size(pepsTensors);

    allProjectors = [computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    projsUL = [proj[1] for proj in allProjectors];
    projsUR = [proj[2] for proj in allProjectors];
    projsDL = [proj[3] for proj in allProjectors];
    projsDR = [proj[4] for proj in allProjectors];
    # projsUL = pepsUnitCell(Lx, Ly, projsUL, unitCellLayout);
    # projsUR = pepsUnitCell(Lx, Ly, projsUR, unitCellLayout);
    # projsDL = pepsUnitCell(Lx, Ly, projsDL, unitCellLayout);
    # projsDR = pepsUnitCell(Lx, Ly, projsDR, unitCellLayout);

    # (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    allTensorsU = [absorption_U(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    nC1 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
    nT1 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
    nC2 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
    # nC1 = pepsUnitCell(Lx, Ly, nC1, unitCellLayout);
    # nT1 = pepsUnitCell(Lx, Ly, nT1, unitCellLayout);
    # nC2 = pepsUnitCell(Lx, Ly, nC2, unitCellLayout);

    allTensorsR = [absorption_D(T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, projsDL, projsDR, (lx, ly)) for lx = 1 : Lx, ly = 1 : Ly];
    nC3 = [allTensorsR[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
    nT3 = [allTensorsR[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
    nC4 = [allTensorsR[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
    # nC3 = pepsUnitCell(Lx, Ly, nC3, unitCellLayout);
    # nT3 = pepsUnitCell(Lx, Ly, nT3, unitCellLayout);
    # nC4 = pepsUnitCell(Lx, Ly, nC4, unitCellLayout);

    return nC1, nT1, nC2, nC3, nT3, nC4

end


function absorption_L(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC4 = ein"(cd, cafe), defb -> ab"(C4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nT4 = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUL[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC1 = ein"adef, (dc, cefb) -> ab"(projsDL[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC4 /= norm(nC4);
    nT4 /= norm(nT4);
    nC1 /= norm(nC1);

    return nC4, nT4, nC1

end

function absorption_L2(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC4 = ein"(cd, cafe), defb -> ab"(C4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)[1]]);
    nT4 = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)[1]], T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUL[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)[1]], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC1 = ein"adef, (dc, cefb) -> ab"(projsDL[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)[1]], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC4 /= norm(nC4);
    nT4 /= norm(nT4);
    nC1 /= norm(nC1);

    return nC4, nT4, nC1

end

function absorption_R(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC2 = ein"(adec, cf), fedb -> ab"(T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsDR[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)...]);
    nT2 = ein"(fgjc, (kefh, bgmei), dhil), ajmkl -> abcd"(projsDR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC3 = ein"bdef, (acef, cd) -> ab"(projsUR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC2 /= norm(nC2);
    nT2 /= norm(nT2);
    nC3 /= norm(nC3);

    return nC2, nT2, nC3
end

function absorption_R2(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC2 = ein"(adec, cf), fedb -> ab"(T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsDR[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)[1]]);
    nT2 = ein"(fgjc, (kefh, bgmei), dhil), ajmkl -> abcd"(projsDR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)[1]], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)[1]], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC3 = ein"bdef, (acef, cd) -> ab"(projsUR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)[1]], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC2 /= norm(nC2);
    nT2 /= norm(nT2);
    nC3 /= norm(nC3);

    return nC2, nT2, nC3
end

function absorption_U(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC1 = ein"(afec, cd), defb -> ab"(T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...]);
    nT1 = ein"(afjg, (fleh, gcmie), hkid), jbmkl -> abcd"(projsUL[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)...], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC2 = ein"adef, (dc, efbc) -> ab"(projsUL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC1 /= norm(nC1);
    nT1 /= norm(nT1);
    nC2 /= norm(nC2);

    return nC1, nT1, nC2

end

function absorption_U2(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC1 = ein"(afec, cd), defb -> ab"(T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUR[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)[2]]);
    nT1 = ein"(afjg, (fleh, gcmie), hkid), jbmkl -> abcd"(projsUL[getCoordinates(x + 0, Lx, y - 1, Ly, unitCellLayout)[2]], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)[2]], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC2 = ein"adef, (dc, efbc) -> ab"(projsUL[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)[2]], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC1 /= norm(nC1);
    nT1 /= norm(nT1);
    nC2 /= norm(nC2);

    return nC1, nT1, nC2

end

function absorption_D(T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, projsDL, projsDR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    
    nC3 = ein"defa, (fecb, dc) -> ab"(projsDL[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nT3 = ein"(fgja, (fhek, gemic), bhil), jkmld -> abcd"(projsDL[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)...], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsDR[getCoordinates(x - 1, Lx, y + 0, Ly, unitCellLayout)...], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC4 = ein"(dc, cefb), adef -> ab"(C4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsDR[getCoordinates(x - 1, Lx, y - 1, Ly, unitCellLayout)...]);

    nC3 /= norm(nC3);
    nT3 /= norm(nT3);
    nC4 /= norm(nC4);

    return nC3, nT3, nC4

end





# function absorption_L!(nC4, nT4, nC1, C1, T1, T3, C4, T4, pepsTensors, projsUL, projsDL, (x, y))
    
#     nC4[x + 0, y + 1] = ein"(cd, cafe), defb -> ab"(C4[x + 0, y + 0], T3[x + 0, y + 0], projsUL[x + 0, y + 0]);
#     nT4[x + 0, y + 1] = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[x + 0, y + 0], T4[x + 0, y + 0], conj(pepsTensors[x + 0, y + 0]), projsUL[x - 1, y + 0], pepsTensors[x + 0, y + 0]);
#     nC1[x + 0, y + 1] = ein"adef, (dc, cefb) -> ab"(projsDL[x - 1, y + 0], C1[x + 0, y + 0], T1[x + 0, y + 0]);

# end

# function absorption_R!(nC2, nT2, nC3, T1, C2, T2, C3, T3, pepsTensors, projsUR, projsDR, (x, y))
    
#     nC2[x + 0, y - 1] = ein"(adec, cf), fedb -> ab"(T1[x + 0, y + 0], C2[x + 0, y + 0], projsDR[x - 1, y - 1]);
#     nT2[x + 0, y - 1] = ein"(fgjc, (kefh, bgmei), dhil), ajmkl -> abcd"(projsDR[x + 0, y - 1], T2[x + 0, y + 0], conj(pepsTensors[x + 0, y + 0]), projsUR[x - 1, y - 1], pepsTensors[x + 0, y + 0]);
#     nC3[x + 0, y - 1] = ein"bdef, (acef, cd) -> ab"(projsUR[x + 0, y - 1], T3[x + 0, y + 0], C3[x + 0, y + 0]);

# end

# function absorption_U!(nC1, nT1, nC2, T4, C1, T1, C2, T2, pepsTensors, projsUL, projsUR, (x, y))
    
#     nC1[x + 1, y + 0] = ein"(afec, cd), defb -> ab"(T4[x + 0, y + 0], C1[x + 0, y + 0], projsUR[x + 0, y - 1]);
#     nT1[x + 1, y + 0] = ein"(afjg, (fleh, gcmie), hkid), jbmkl -> abcd"(projsUL[x + 0, y - 1], T1[x + 0, y + 0], conj(pepsTensors[x + 0, y + 0]), projsUR[x + 0, y + 0], pepsTensors[x + 0, y + 0]);
#     nC2[x + 1, y + 0] = ein"adef, (dc, efbc) -> ab"(projsUL[x + 0, y + 0], C2[x + 0, y + 0], T2[x + 0, y + 0]);

#     return nC1, nT1, nC2

# end

# function absorption_D!(nC3, nT3, nC4, T2, C3, T3, C4, T4, pepsTensors, projsDL, projsDR, (x, y))
    
#     nC3[x - 1, y + 0] = ein"defa, (fecb, dc) -> ab"(projsDL[x - 1, y + 0], T2[x + 0, y + 0], C3[x + 0, y + 0]);
#     nT3[x - 1, y + 0] = ein"(fgja, (fhek, gemic), bhil), jkmld -> abcd"(projsDL[x - 1, y - 1], T3[x + 0, y + 0], conj(pepsTensors[x + 0, y + 0]), projsDR[x - 1, y + 0], pepsTensors[x + 0, y + 0]);
#     nC4[x - 1, y + 0] = ein"(dc, cefb), adef -> ab"(C4[x + 0, y + 0], T4[x + 0, y + 0], projsDR[x - 1, y - 1]);
    
#     return nC3, nT3, nC4

# end