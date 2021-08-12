
function absorptionStep_L2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in y-direction
    for ucLy = 1 : +1 : unitCellLy

        # compute projectore for one column
        allProjectors = [computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy)) for ucLx = 1 : unitCellLx];
        projsUL = [proj[1] for proj in allProjectors];
        projsDL = [proj[2] for proj in allProjectors];
        # projsUR = [proj[3] for proj in allProjectors];
        # projsDR = [proj[4] for proj in allProjectors];

        # make absorption of one column
        absorbedTensorsL = [absorption_L2(C1, T1, T3, C4, T4, pepsTensors, unitCellLayout, projsUL, projsDL, (ucLx, ucLy)) for ucLx = 1 : unitCellLx];

        # set updated tensors (please)
        C4 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] ? absorbedTensorsL[lx][1] : C4[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        T4 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] ? absorbedTensorsL[lx][2] : T4[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        C1 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy + 1, Ly, unitCellLayout)...] ? absorbedTensorsL[lx][3] : C1[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];

    end

    return C4, T4, C1

end

function absorptionStep_U2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in x-direction
    for ucLx = 1 : +1 : unitCellLx

        # compute projectore for one row
        allProjectors = [computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy)) for ucLy = 1 : unitCellLy];
        projsUL = [proj[1] for proj in allProjectors];
        projsUR = [proj[2] for proj in allProjectors];
        # projsDL = [proj[3] for proj in allProjectors];
        # projsDR = [proj[4] for proj in allProjectors];

        # make absorption of one row
        absorbedTensorsU = [absorption_U2(T4, C1, T1, C2, T2, pepsTensors, unitCellLayout, projsUL, projsUR, (ucLx, ucLy)) for ucLy = 1 : unitCellLy];

        # set updated tensors (please)
        C1 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsU[ly][1] : C1[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        T1 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsU[ly][2] : T1[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        C2 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsU[ly][3] : C2[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];

        # # get updated tensors
        # C1 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
        # T1 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
        # C2 = [allTensorsU[getCoordinates(idx + 1, Lx, idy + 0, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
        
    end

    return C1, T1, C2

end

function absorptionStep_R2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in y-direction
    for ucLy = unitCellLy : -1 : 1

        # compute projectore for one column
        allProjectors = [computeIsometries_LR(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy)) for ucLx = 1 : unitCellLx];
        # projsUL = hcat(projsUL, [proj[1] for proj in allProjectors]);
        # projsDL = hcat(projsDL, [proj[2] for proj in allProjectors]);
        projsUR = [proj[3] for proj in allProjectors];
        projsDR = [proj[4] for proj in allProjectors];

        # make absorption of one column
        absorbedTensorsR = [absorption_R2(T1, C2, T2, C3, T3, pepsTensors, unitCellLayout, projsUR, projsDR, (ucLx, ucLy)) for ucLx = 1 : unitCellLx];

        # set updated tensors
        C2 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] ? absorbedTensorsR[lx][1] : C2[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        T2 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] ? absorbedTensorsR[lx][2] : T2[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        C3 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(lx + 0, Lx, ucLy - 1, Ly, unitCellLayout)...] ? absorbedTensorsR[lx][3] : C3[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];

        # # get updated tensors
        # C2 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
        # T2 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
        # C3 = [allTensorsR[getCoordinates(idx + 0, Lx, idy - 1, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
        
    end

    return C2, T2, C3

end

function absorptionStep_D2(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE)

    # get size of unit cell
    Lx, Ly = size(pepsTensors);
    unitCellLx, unitCellLy = size(unitCellLayout);

    # loop over all tensor in x-direction
    for ucLx = unitCellLx : -1 : unitCellLx

        # compute projectore for one row
        allProjectors = [computeIsometries_UD(C1, T1, C2, T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, chiE, truncBelowE, (ucLx, ucLy)) for ucLy = 1 : unitCellLy];
        # projsUL = [proj[1] for proj in allProjectors];
        # projsUR = [proj[2] for proj in allProjectors];
        projsDL = [proj[3] for proj in allProjectors];
        projsDR = [proj[4] for proj in allProjectors];

        # make absorption of one row
        absorbedTensorsD = [absorption_D(T2, C3, T3, C4, T4, pepsTensors, unitCellLayout, projsDL, projsDR, (ucLx, ucLy)) for ucLy = 1 : unitCellLy];

        # set updated tensors (please)
        C3 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsD[ly][1] : C3[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        T3 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsD[ly][2] : T3[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];
        C4 = [toTensorNum(lx, Lx, ly, Ly) == unitCellLayout[getCoordinates(ucLx + 1, Lx, ly + 0, Ly, unitCellLayout)...] ? absorbedTensorsD[ly][3] : C4[lx, ly] for lx = 1 : Lx, ly = 1 : Ly];

        # # get updated tensors
        # C3 = [allTensorsD[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][1] for idx = 1 : Lx, idy = 1 : Ly];
        # T3 = [allTensorsD[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][2] for idx = 1 : Lx, idy = 1 : Ly];
        # C4 = [allTensorsD[getCoordinates(idx - 1, Lx, idy + 0, Ly, unitCellLayout)...][3] for idx = 1 : Lx, idy = 1 : Ly];
        
    end

    return C3, T3, C4

end


# function performAbsorption_CTM(gammaTensors, unitCell, environmentTensors, chiE, truncBelow = 1e-10)

#     # # get size of the system
#     # Nx, Ny = size(gammaTensors);

#     # get size of the actual unitCell
#     unitCellLx, unitCellLy = size(unitCell);

#     # run over all tensors in the y direction
#     for unitCellIdy = 1 : unitCellLy

#         # determine permutation of the unit cell
#         permIdy = mod.(collect((unitCellIdy - 1) : (unitCellIdy + unitCellLy - 2)),unitCellLy) .+ 1;

#         # permute unitCell according to current CTM step
#         unitCell = unitCell[:,permIdy];


#         #----------------------------------------------------------------------
#         # compute projectors to truncate environment bond dimension
#         #----------------------------------------------------------------------

#         # loop over all tensors in the x direction to compute projectors
#         listOfProjectors_L = Vector{Any}(undef, unitCellLx + 1);
#         listOfProjectors_R = Vector{Any}(undef, unitCellLx + 1);
#         for unitCellIdx = 1 : unitCellLx

#             # determine permutation of the unit cell
#             permIdx = mod.(collect((unitCellIdx - 1) : (unitCellIdx + unitCellLx - 2)),unitCellLx) .+ 1;

#             # permute unitCell according to current CTM step
#             unitCell = unitCell[permIdx,:];

#             # select relevant gammaTensors, conjGammaTensors and environmentTensors
#             relevantPartUnitCell = unitCell[1 : 2,1 : 2];
#             relevantGammaTensors = gammaTensors[relevantPartUnitCell];
#             relevantEnvironmentTensors = environmentTensors[relevantPartUnitCell];


#             #------------------------------------------------------------------
#             # contract upper half of the network
#             #------------------------------------------------------------------

#             envTensorsU = Vector{TensorMap}(undef, 6);
#             envTensorsU[1] = relevantEnvironmentTensors[1,1][8];
#             envTensorsU[2] = relevantEnvironmentTensors[1,1][1];
#             envTensorsU[3] = relevantEnvironmentTensors[1,1][2];
#             envTensorsU[4] = relevantEnvironmentTensors[1,2][2];
#             envTensorsU[5] = relevantEnvironmentTensors[1,2][3];
#             envTensorsU[6] = relevantEnvironmentTensors[1,2][4];

#             # contract upper half and reshape rhoU into matrix
#             @tensor rhoUL[-1 -2 -3; -4 -5 -6] := envTensorsU[1][-1 3 5 1] * envTensorsU[2][1 2] * envTensorsU[3][2 6 4 -6] * relevantGammaTensors[1,1][5 -2 7 -5 6] * conj(relevantGammaTensors[1,1][3 -3 7 -4 4]);
#             @tensor rhoUR[-1 -2 -3 -4 -5 -6; ()] := envTensorsU[4][-1 6 4 2] * envTensorsU[5][2 1] * envTensorsU[6][5 3 -6 1] * relevantGammaTensors[1,2][-2 -4 7 5 6] * conj(relevantGammaTensors[1,2][-3 -5 7 3 4]);
#             @tensor rhoU[-1 -2 -3 -4 -5 -6; ()] := rhoUL[-1 -2 -3 3 2 1] * rhoUR[1 2 3 -4 -5 -6];
#             rhoU /= norm(rhoU);
#             # println(space(rhoU))

#             # perform SVD of rhoU
#             # UU, SU, VU = tsvd(rhoU, (1,2,3), (4,5,6));
#             UU, SU, VU = tsvd(rhoU, (1,2,3), (4,5,6), trunc = truncbelow(truncBelow));
#             # UU, SU, VU = tsvd(rhoU, (1,2,3), (4,5,6), trunc = truncdim(chiE));
#             # UU, SU, VU = tsvd(UU * SU * VU, (1,2,3), (4,5,6), trunc = truncbelow(1e-12));
#             # println(UU)
#             # println(SU)
#             # println(VU)

#             # absorb sqrt(SU) into UU and VU
#             FUL = permute(UU * sqrt(SU), (1,2,3), (4,));
#             FUR = permute(sqrt(SU) * VU, (2,3,4), (1,));
#             # println(FUL)
#             # println(FUR)


#             #------------------------------------------------------------------
#             # contract lower half of the network
#             #------------------------------------------------------------------

#             envTensorsD = Vector{Any}(undef, 6);
#             envTensorsD[1] = relevantEnvironmentTensors[2,1][8];
#             envTensorsD[2] = relevantEnvironmentTensors[2,1][7];
#             envTensorsD[3] = relevantEnvironmentTensors[2,1][6];
#             envTensorsD[4] = relevantEnvironmentTensors[2,2][6];
#             envTensorsD[5] = relevantEnvironmentTensors[2,2][5];
#             envTensorsD[6] = relevantEnvironmentTensors[2,2][4];

#             # contract upper half and reshape rhoD into matrix
#             @tensor rhoDL[(); -1 -2 -3 -4 -5 -6] := envTensorsD[1][2 4 6 -6] * envTensorsD[2][1 2] * envTensorsD[3][1 -1 3 5] * relevantGammaTensors[2,1][6 5 7 -3 -5] * conj(relevantGammaTensors[2,1][4 3 7 -2 -4]);
#             @tensor rhoDR[-1 -2 -3; -4 -5 -6] := envTensorsD[4][-3 1 3 5] * envTensorsD[5][1 2] * envTensorsD[6][6 4 2 -4] * relevantGammaTensors[2,2][-1 5 7 6 -6] * conj(relevantGammaTensors[2,2][-2 3 7 4 -5]);
#             @tensor rhoD[(); -1 -2 -3 -4 -5 -6] := rhoDR[3 2 1 -1 -2 -3] * rhoDL[1 2 3 -4 -5 -6];
#             rhoD /= norm(rhoD);
#             # println(space(rhoD))

#             # perform SVD of rhoD
#             # UD, SD, VD = tsvd(rhoD, (1,2,3), (4,5,6));
#             UD, SD, VD = tsvd(rhoD, (1,2,3), (4,5,6), trunc = truncbelow(truncBelow));
#             # UD, SD, VD = tsvd(rhoD, (1,2,3), (4,5,6), trunc = truncdim(chiE));
#             # UD, SD, VD = tsvd(UD * SD * VD, (1,2,3), (4,5,6), trunc = truncbelow(1e-12));
#             # println(UD)
#             # println(SD)
#             # println(VD)

#             # absorb sqrt(SD) into UD and VD
#             FDR = permute(UD * sqrt(SD), (4,), (1,2,3));
#             FDL = permute(sqrt(SD) * VD, (1,), (2,3,4));
#             # println(FDR)
#             # println(FDL)


#             #------------------------------------------------------------------
#             # compute projectors for the left side
#             #------------------------------------------------------------------

#             # compute biorthogonalization of FUL and FDL tensors and truncate UL, SL and VL
#             @tensor BOL[-1; -2] := FDL[-1 3 2 1] * FUL[1 2 3 -2];
#             BOL /= norm(BOL);
#             UL, SL, VL = tsvd(BOL, (1,), (2,), trunc = truncdim(chiE), alg = TensorKit.SVD());
#             UL, SL, VL = tsvd(UL * SL * VL, (1,), (2,), trunc = truncbelow(truncBelow), alg = TensorKit.SVD());
#             sqrtSL = pinv(sqrt(SL));

#             # build projectors for left truncation
#             PUL = FUL * VL' * sqrtSL;
#             PDL = sqrtSL * UL' * FDL;
#             # @tensor PUL[-1 -2; -3] = FUL[-1 -2 1] * VL'[1 2] * sqrtSL[2 -3];
#             # @tensor PDL[(); -1 -2 -3] = FDL[1 -2 -3] * UL'[1 2] * sqrtSL[2 -1];
#             # println(PUL)
#             # println(PDL)

#             # # repeat biorthogonalization procedure
#             # for idxB = 1 : 3
#             #     oldPUL = PUL;
#             #     oldPDL = PDL;

#             #     @tensor BOL[-1; -2] := oldPDL[-1 3 2 1] * oldPUL[1 2 3 -2];
#             #     UL, SL, VL = tsvd(BOL, (1,), (2,), trunc = truncdim(chiE), alg = TensorKit.SVD());
#             #     UL, SL, VL = tsvd(UL * SL * VL, (1,), (2,), trunc = truncbelow(truncBelow), alg = TensorKit.SVD());
#             #     sqrtSL = pinv(sqrt(SL));

#             #     PUL = oldPUL * VL' * sqrtSL;
#             #     PDL = sqrtSL * UL' * oldPDL;
#             # end

#             # store projectors
#             listOfProjectors_L[unitCellIdx + 1] = [PDL , PUL];


#             #------------------------------------------------------------------
#             # compute projectors for the right side
#             #------------------------------------------------------------------

#             # compute biorthogonalization of FUR and FDR tensors
#             @tensor BOR[-1; -2] := FDR[-1 1 2 3] * FUR[3 2 1 -2];
#             BOR /= norm(BOR);
#             UR, SR, VR = tsvd(BOR, (1,), (2,), trunc = truncdim(chiE), alg = TensorKit.SVD());
#             UR, SR, VR = tsvd(UR * SR * VR, (1,), (2,), trunc = truncbelow(truncBelow), alg = TensorKit.SVD());
#             sqrtSR = pinv(sqrt(SR));

#             # build projectors for right truncation
#             PUR = FUR * VR' * sqrtSR;
#             PDR = sqrtSR * UR' * FDR;
#             # @tensor PUR[-1 -2 -3; ()] = sqrtSR[-1 1] * UR'[1 2] * FUR[2 -2 -3];
#             # @tensor PDR[-1; -2 -3] = sqrtSR[-1 1] * VR'[1 2] * FDR[2 -2 -3];
#             # println(PUR)
#             # println(PDR)

#             # # repeat biorthogonalization procedure
#             # for idxB = 1 : 3
#             #     oldPUR = PUR;
#             #     oldPDR = PDR;

#             #     @tensor BOR[-1; -2] := oldPDR[-1 1 2 3] * oldPUR[3 2 1 -2];
#             #     UR, SR, VR = tsvd(BOR, (1,), (2,), trunc = truncdim(chiE), alg = TensorKit.SVD());
#             #     UR, SR, VR = tsvd(UR * SR * VR, (1,), (2,), trunc = truncbelow(truncBelow), alg = TensorKit.SVD());
#             #     sqrtSR = pinv(sqrt(SR));

#             #     PUR = oldPUR * VR' * sqrtSR;
#             #     PDR = sqrtSR * UR' * oldPDR;
#             # end

#             # store projectors
#             listOfProjectors_R[unitCellIdx + 1] = [PDR , PUR];


#             # permute unitCell back
#             permIdxR = sortperm(permIdx);
#             unitCell = unitCell[permIdxR,:];

#         end

#         # store last set of projectors also to the first position because the subspaces must be the same
#         listOfProjectors_L[1] = listOfProjectors_L[unitCellLx + 1];
#         listOfProjectors_R[1] = listOfProjectors_R[unitCellLx + 1];


#         #----------------------------------------------------------------------
#         # make absorption step towards the left
#         #----------------------------------------------------------------------

#         # copy environmentTensors to avoid using already updated tensors
#         oldEnvironmentTensors = deepcopy(environmentTensors);

#         # loop over all tensors in the x direction
#         for unitCellIdx = 1 : unitCellLx

#             # create and truncate new top corner tensor
#             truncationProjectorD = listOfProjectors_L[unitCellIdx + 0][1];
#             @tensor nC[-1; -2] := oldEnvironmentTensors[unitCell[unitCellIdx,1]][1][2 1] * oldEnvironmentTensors[unitCell[unitCellIdx,1]][2][1 3 4 -2] * truncationProjectorD[-1 4 3 2];
#             # println(nC)
#             # println(space(nC))

#             # normalize and store tensor
#             nC /= norm(nC);
#             environmentTensors[unitCell[unitCellIdx,2]][1] = nC;

#             # create and truncate new line tensor
#             truncationProjectorU = listOfProjectors_L[unitCellIdx + 0][2];
#             truncationProjectorD = listOfProjectors_L[unitCellIdx + 1][1];
#             @tensor nT[-1; -2 -3 -4] := truncationProjectorD[-1 2 4 1] * oldEnvironmentTensors[unitCell[unitCellIdx,1]][8][1 3 5 7] * gammaTensors[unitCell[unitCellIdx,1]][5 4 6 -3 9] * conj(gammaTensors[unitCell[unitCellIdx,1]][3 2 6 -2 8]) * truncationProjectorU[7 9 8 -4];
#             # println(nT)
#             # println(space(nT))

#             # normalize and store tensor
#             nT /= norm(nT);
#             environmentTensors[unitCell[unitCellIdx,2]][8] = nT;

#             # create and truncate new bottom corner tensor
#             truncationProjectorU = listOfProjectors_L[unitCellIdx + 1][2];
#             @tensor nC[(); -1 -2] := oldEnvironmentTensors[unitCell[unitCellIdx,1]][6][1 -1 4 3] * oldEnvironmentTensors[unitCell[unitCellIdx,1]][7][1 2] * truncationProjectorU[2 3 4 -2];
#             # println(nC)
#             # println(space(nC))

#             # normalize and store tensor
#             nC /= norm(nC);
#             environmentTensors[unitCell[unitCellIdx,2]][7] = nC;

#         end


#         # #----------------------------------------------------------------------
#         # # make absorption step towards the right
#         # #----------------------------------------------------------------------

#         # # copy environmentTensors to avoid using already updated tensors
#         # oldEnvironmentTensors = deepcopy(environmentTensors);

#         # # loop over all tensors in the x direction
#         # for unitCellIdx = 1 : unitCellLx

#         #     # create and truncate new top corner tensor
#         #     truncationProjectorD = listOfProjectors_R[unitCellIdx + 0][1];
#         #     @tensor nC[-1 -2; ()] := oldEnvironmentTensors[unitCell[unitCellIdx,2]][2][-1 4 3 1] * oldEnvironmentTensors[unitCell[unitCellIdx,2]][3][1 2] * truncationProjectorD[-2 2 3 4];
#         #     # println(nC)
#         #     # println(space(nC))

#         #     # normalize and store tensor
#         #     nC /= norm(nC);
#         #     environmentTensors[unitCell[unitCellIdx,1]][3] = nC;


#         #     # create and truncate new line tensor
#         #     truncationProjectorU = listOfProjectors_R[unitCellIdx + 0][2];
#         #     truncationProjectorD = listOfProjectors_R[unitCellIdx + 1][1];
#         #     @tensor nT[-1 -2 -3; -4] := truncationProjectorD[-3 1 2 4] * oldEnvironmentTensors[unitCell[unitCellIdx,2]][4][5 3 1 7] * gammaTensors[unitCell[unitCellIdx,2]][-1 4 6 5 9] * conj(gammaTensors[unitCell[unitCellIdx,2]][-2 2 6 3 8]) * truncationProjectorU[9 8 7 -4];
#         #     # println(nT)
#         #     # println(space(nT))

#         #     # normalize and store tensor
#         #     nT /= norm(nT);
#         #     environmentTensors[unitCell[unitCellIdx,1]][4] = nT;

            
#         #     # create and truncate new bottom corner tensor
#         #     truncationProjectorU = listOfProjectors_R[unitCellIdx + 1][2];
#         #     @tensor nC[-1; -2] := oldEnvironmentTensors[unitCell[unitCellIdx,2]][6][-1 1 3 4] * oldEnvironmentTensors[unitCell[unitCellIdx,2]][5][1 2] * truncationProjectorU[4 3 2 -2];
#         #     # println(nC)
#         #     # println(space(nC))

#         #     # normalize and store tensor
#         #     nC /= norm(nC);
#         #     environmentTensors[unitCell[unitCellIdx,1]][5] = nC;

#         # end

#         # permute unitCell back
#         permIdyR = sortperm(permIdy);
#         unitCell = unitCell[:,permIdyR];

#     end

#     # function return
#     return environmentTensors;

# end