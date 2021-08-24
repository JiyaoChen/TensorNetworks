
function periodicIndex(idx, arrayLength)
    return mod(idx - 1, arrayLength) + 1;
end

function absorptionStep_L(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensors in y-direction
    for ucLy = 1 : +1 : unitCellLy

        for ucLx = 1 : unitCellLx

            allProjectors = [computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy), 'L') for ucLx = 1 : unitCellLx];
            projsUL = [proj[1] for proj in allProjectors];
            projsDL = [proj[2] for proj in allProjectors];

            nC4, nT4, nC1 = absorption_L(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (ucLx, ucLy));

            C4[getCoordinates(ucLx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] = nC4;
            T4[getCoordinates(ucLx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] = nT4;
            C1[getCoordinates(ucLx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] = nC1;

        end

    end

    return C4, T4, C1

end

function absorptionStep_U(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensors in x-direction
    for ucLx = 1 : +1 : unitCellLx

        for ucLy = 1 : unitCellLy

            allProjectors = [computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy), 'U') for ucLy = 1 : unitCellLy];
            projsUL = [proj[1] for proj in allProjectors];
            projsUR = [proj[2] for proj in allProjectors];

            nC1, nT1, nC2 = absorption_U(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (ucLx, ucLy));

            C1[getCoordinates(ucLx + 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nC1;
            T1[getCoordinates(ucLx + 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nT1;
            C2[getCoordinates(ucLx + 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nC2;

        end

    end

    return C1, T1, C2

end

function absorptionStep_R(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in y-direction
    for ucLy = unitCellLy : -1 : 1

        for ucLx = 1 : unitCellLx

            allProjectors = [computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy), 'R') for ucLx = 1 : unitCellLx];
            projsUR = [proj[3] for proj in allProjectors];
            projsDR = [proj[4] for proj in allProjectors];

            nC2, nT2, nC3 = absorption_R(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (ucLx, ucLy));

            C2[getCoordinates(ucLx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] = nC2;
            T2[getCoordinates(ucLx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] = nT2;
            C3[getCoordinates(ucLx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] = nC3;

        end

    end

    return C2, T2, C3

end

function absorptionStep_D(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in x-direction
    for ucLx = unitCellLx : -1 : 1

        for ucLy = 1 : unitCellLy

            allProjectors = [computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy), 'D') for ucLy = 1 : unitCellLy];
            projsDL = [proj[3] for proj in allProjectors];
            projsDR = [proj[4] for proj in allProjectors];

            # make absorption of one row
            nC3, nT3, nC4 = absorption_D(T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, projsDL, projsDR, (ucLx, ucLy));

            # set updated tensors (please)
            C3[getCoordinates(ucLx - 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nC3;
            T3[getCoordinates(ucLx - 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nT3;
            C4[getCoordinates(ucLx - 1, Lx, ucLy + 0, Ly, unitCellLayout)...] = nC4;

        end

    end

    return C3, T3, C4

end

function absorption_U(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(pepsTensors);
    
    nC1 = ein"(afec, cd), defb -> ab"(T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUR[periodicIndex(y - 1, unitCellLy)]);
    nT1 = ein"(afjg, (fleh, gcmie), hkid), jbmkl -> abcd"(projsUL[periodicIndex(y - 1, unitCellLy)], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[periodicIndex(y + 0, unitCellLy)], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC2 = ein"adef, (dc, efbc) -> ab"(projsUL[periodicIndex(y + 0, unitCellLy)], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC1 /= norm(nC1);
    nT1 /= norm(nT1);
    nC2 /= norm(nC2);

    return nC1, nT1, nC2

end

function absorption_L(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);
    
    nC4 = ein"(cd, cafe), defb -> ab"(C4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsUL[periodicIndex(x + 0, unitCellLx)]);
    nT4 = ein"(afkg, (fejh, egmbi), hlid), jkmcl -> abcd"(projsDL[periodicIndex(x + 0, unitCellLx)], T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUL[periodicIndex(x - 1, unitCellLx)], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC1 = ein"adef, (dc, cefb) -> ab"(projsDL[periodicIndex(x - 1, unitCellLx)], C1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC4 /= norm(nC4);
    nT4 /= norm(nT4);
    nC1 /= norm(nC1);

    return nC4, nT4, nC1

end

function absorption_D(T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, projsDL, projsDR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(pepsTensors);
    
    nC3 = ein"defa, (fecb, dc) -> ab"(projsDL[periodicIndex(y + 0, unitCellLy)], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nT3 = ein"(fgja, (fhek, gemic), bhil), jkmld -> abcd"(projsDL[periodicIndex(y - 1, unitCellLy)], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsDR[periodicIndex(y + 0, unitCellLy)], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC4 = ein"(dc, cefb), adef -> ab"(C4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], T4[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsDR[periodicIndex(y - 1, unitCellLy)]);

    nC3 /= norm(nC3);
    nT3 /= norm(nT3);
    nC4 /= norm(nC4);

    return nC3, nT3, nC4

end

function absorption_R(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (x, y))

    # get size
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);
    
    nC2 = ein"(adec, cf), fedb -> ab"(T1[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], projsDR[periodicIndex(x - 1, unitCellLx)]);
    nT2 = ein"(fgjc, (kefh, bgmei), dhil), ajmkl -> abcd"(projsDR[periodicIndex(x + 0, unitCellLx)], T2[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], conj(pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]), projsUR[periodicIndex(x - 1, unitCellLx)], pepsTensors[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);
    nC3 = ein"bdef, (acef, cd) -> ab"(projsUR[periodicIndex(x + 0, unitCellLx)], T3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...], C3[getCoordinates(x + 0, Lx, y + 0, Ly, unitCellLayout)...]);

    nC2 /= norm(nC2);
    nT2 /= norm(nT2);
    nC3 /= norm(nC3);

    return nC2, nT2, nC3

end