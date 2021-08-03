function absorptionStep_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE)
    
    projsUL, projsDL = similar.((T1, T3));
    projsUR, projsDR = similar.((T3, T1));
    nC4, nT4, nC1 = similar.((C4, T4, C1));
    nC2, nT2, nC3 = similar.((C2, T2, C3));

    foreach(keys(iPEPS.tensorDict)) do tensorKey
        projsUL[tensorKey...], projsDL[tensorKey...], projsUR[tensorKey...], projsDR[tensorKey...] = computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE, tensorKey);
    end

    (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    foreach(keys(iPEPS.tensorDict)) do tensorKey
        absorption_L!(nC4, nT4, nC1, oC1, oT1, oT3, oC4, oT4, iPEPS, projsUL, projsDL, tensorKey);
        absorption_R!(nC2, nT2, nC3, oT1, oC2, oT2, oC3, oT3, iPEPS, projsUR, projsDR, tensorKey);
    end

    return nC4, nT4, nC1, nC2, nT2, nC3

end

function absorptionStep_UD(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE)
    
    projsUL, projsUR = similar.((T3, T1));
    projsDL, projsDR = similar.((T1, T3));
    nC1, nT1, nC2 = similar.((C1, T1, C2));
    nC3, nT3, nC4 = similar.((C3, T3, C4));

    tensorKeys = keys(iPEPS.tensorDict)
    # foreach(keys(iPEPS.tensorDict)) do tensorKey
    foreach(tensorKeys) do tensorKey
        projsUL[tensorKey...], projsUR[tensorKey...], projsDL[tensorKey...], projsDR[tensorKey...] = computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, iPEPS, chiE, truncBelowE, tensorKey);
    end

    (oC1, oT1, oC2, oT2, oC3, oT3, oC4, oT4) = deepcopy.((C1, T1, C2, T2, C3, T3, C4, T4));
    foreach(keys(iPEPS.tensorDict)) do tensorKey
        absorption_U!(nC1, nT1, nC2, oT4, oC1, oT1, oC2, oT2, iPEPS, projsUL, projsUR, tensorKey);
        absorption_D!(nC3, nT3, nC4, oT2, oC3, oT3, oC4, oT4, iPEPS, projsDL, projsDR, tensorKey);
    end

    return nC1, nT1, nC2, nC3, nT3, nC4

end

function absorption_L!(nC4, nT4, nC1, C1, T1, T3, C4, T4, iPEPS, projsUL, projsDL, (x, y))
    
    nC4[x + 0, y + 1] = ein"(cd, cafe), defb -> ab"(C4[x + 0, y + 0], T3[x + 0, y + 0], projsUL[x + 0, y + 0]);
    nT4[x + 0, y + 1] = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[x + 0, y + 0], T4[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), projsUL[x - 1, y + 0], iPEPS[x + 0, y + 0]);
    nC1[x + 0, y + 1] = ein"adef, (dc, cefb) -> ab"(projsDL[x - 1, y + 0], C1[x + 0, y + 0], T1[x + 0, y + 0]);

end

function absorption_R!(nC2, nT2, nC3, T1, C2, T2, C3, T3, iPEPS, projsUR, projsDR, (x, y))
    
    nC2[x + 0, y - 1] = ein"(adec, cf), fedb -> ab"(T1[x + 0, y + 0], C2[x + 0, y + 0], projsDR[x - 1, y - 1]);
    nT2[x + 0, y - 1] = ein"(fgjc, (kefh, bgmei), dhil), ajmkl -> abcd"(projsDR[x + 0, y - 1], T2[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), projsUR[x - 1, y - 1], iPEPS[x + 0, y + 0]);
    nC3[x + 0, y - 1] = ein"bdef, (acef, cd) -> ab"(projsUR[x + 0, y - 1], T3[x + 0, y + 0], C3[x + 0, y + 0]);

end

function absorption_U!(nC1, nT1, nC2, T4, C1, T1, C2, T2, iPEPS, projsUL, projsUR, (x, y))
    
    nC1[x - 1, y + 0] = ein"(afec, cd), defb -> ab"(T4[x + 0, y + 0], C1[x + 0, y + 0], projsUR[x + 0, y - 1]);
    nT1[x - 1, y + 0] = ein"(afjg, (fleh, gcmie), hkid), jbmkl -> abcd"(projsUL[x + 0, y - 1], T1[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), projsUR[x + 0, y + 0], iPEPS[x + 0, y + 0]);
    nC2[x - 1, y + 0] = ein"adef, (dc, efbc) -> ab"(projsUL[x + 0, y + 0], C2[x + 0, y + 0], T2[x + 0, y + 0]);

end

function absorption_D!(nC3, nT3, nC4, T2, C3, T3, C4, T4, iPEPS, projsDL, projsDR, (x, y))
    
    nC3[x - 1, y + 0] = ein"defa, (fecb, dc) -> ab"(projsDL[x - 1, y + 0], T2[x + 0, y + 0], C3[x + 0, y + 0]);
    nT3[x - 1, y + 0] = ein"(fgja, (fhek, gemic), bhil), jkmld -> abcd"(projsDL[x - 1, y - 1], T3[x + 0, y + 0], conj(iPEPS[x + 0, y + 0]), projsDR[x - 1, y + 0], iPEPS[x + 0, y + 0]);
    nC4[x - 1, y + 0] = ein"(dc, cefb), adef -> ab"(C4[x + 0, y + 0], T4[x + 0, y + 0], projsDR[x - 1, y - 1]);

end