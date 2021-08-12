
function arbitraryUnitCell_SquareLattice(Lx, Ly)

    # number of distinct links in the iPEPS network
    numLinks = 2 * Lx * Ly;

    # find ID of lambda matrices for all the tensors in the Lx x Ly unit cell
    lambdaIndexTable = zeros(Int64, Lx * Ly, 6);
    idx = 1;
    if  any([Lx, Ly] .== 1)
        Lxy = maximum([Lx, Ly]);
        for y = 1 : Ly
            for x = 1 : Lx
                lambdaIndexTable[idx,:] = [x , y , mod(x + (y - 1) - 1, Lxy) + 1 + Lxy , mod(x + (y - 1) - 0, Lxy) + 1 , mod(x + (y - 1) - 0, Lxy) + 1 + Lxy , mod(x + (y - 1) - 1, Lxy) + 1 ];
                idx = idx + 1;
            end
        end
    else
        for y = 1 : Ly
            for x = 1 : Lx
                lambdaIndexTable[idx,:] = [x , y , y + Ly + 2 * Ly * (x - 1) , y + 2 * Ly * mod(x, Lx) , 1 + Ly + 2 * Ly * (x - 1) + mod(y, Ly) , y + 2 * Ly * (x - 1)];
                idx = idx + 1;
            end
        end
    end
    # println(lambdaIndexTable)

    # get linear indices for lambdaIndexTable
    linInds = LinearIndices(lambdaIndexTable[:,3 : 6]);
    
    # find which tensors and which indices are involved for every lambda matrix
    lambdaUpdateTensors = zeros(Int64, numLinks, 5);
    for lambdaMatrix = 1 : numLinks
        
        # find the tensors connected to lambdaMatrix
        bitVec = sum(lambdaIndexTable[:,3 : 6] .== lambdaMatrix, dims = 2);
        linInds = LinearIndices(bitVec);
        relevantTensors = linInds[findall(x -> x == 1, bitVec)];
        
        # find index number where lambdaMatrix is connected
        bitVecL = lambdaIndexTable[relevantTensors[1], 3 : 6] .== lambdaMatrix;
        bitVecR = lambdaIndexTable[relevantTensors[2], 3 : 6] .== lambdaMatrix;
        linIndsL = LinearIndices(bitVecL);
        linIndsR = LinearIndices(bitVecR);
        posLambdaL = linIndsL[findall(x -> x == 1, bitVecL)];
        posLambdaR = linIndsR[findall(x -> x == 1, bitVecR)];
        
        tensorIndex = [posLambdaL posLambdaR ];
        
        lambdaUpdateTensors[lambdaMatrix,:] = [ lambdaMatrix relevantTensors' tensorIndex];
        
    end
    # println(lambdaUpdateTensors)

    # function return
    return numLinks, lambdaIndexTable, lambdaUpdateTensors;

end

function initialize_iPEPS_UnitCell(Lx::Int64, Ly::Int64, indBondDim::Vector{Int64}, d::Int64, numLinks::Int64, lambdaIndexTable)

    # initialize gammaTensors
    gammaTensors = Array{Any, 2}(undef, Lx, Ly);
    for idx = 1 : Lx
        
        for idy = 1 : Ly
        
            # find lambda tensors attached to iPEPS tensor
            bitVec = (lambdaIndexTable[:, 1] .== idx) .& (lambdaIndexTable[:, 2] .== idy);
            linInds = LinearIndices(bitVec);
            lambdaTensors = lambdaIndexTable[linInds[findall(x -> x == 1, bitVec)], 3 : 6];
            
            # initialize iPEPS tensor with the correct bond dimensions
            gammaTensors[idx, idy] = randn(Float64, indBondDim[lambdaTensors[1]], indBondDim[lambdaTensors[2]], d, indBondDim[lambdaTensors[3]], indBondDim[lambdaTensors[4]]);

        end
        
    end

    # initialize lambdaTensors
    lambdaTensors = Vector{Any}(undef, numLinks);
    for lambdaIdx = 1 : numLinks
        lambdaTensors[lambdaIdx] = Matrix{Float64}(I, indBondDim[lambdaIdx], indBondDim[lambdaIdx]);
    end

    # function return
    return gammaTensors, lambdaTensors;

end

# simple update on horizontal bonds
function simpleUpdate_iPEPS_H(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate)

    # determine gamma and lambda tensors
    posΓL = [lambdaIndexTable[updateLambdaMatrix[2],1], lambdaIndexTable[updateLambdaMatrix[2],2]];
    posΓR = [lambdaIndexTable[updateLambdaMatrix[3],1], lambdaIndexTable[updateLambdaMatrix[3],2]];
    ΓL = gammaTensors[posΓL[1], posΓL[2]];
    ΓR = gammaTensors[posΓR[1], posΓR[2]];
    
    lambdaL1 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],3]];
    lambdaL2 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],4]];
    lambdaL5 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],6]];

    lambdaLR = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],5]];

    lambdaR2 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],4]];
    lambdaR4 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],5]];
    lambdaR5 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],6]];
    
    # absorb all lambdas into ΓL and ΓR
    @ein ΓL[-1, -2, -3, -4, -5] := lambdaL1[-1, 1] * lambdaL2[-2, 2] * ΓL[1, 2, -3, -4, 5] * lambdaL5[5, -5];
    @ein ΓR[-1, -2, -3, -4, -5] := lambdaR2[-2, 2] * ΓR[-1, 2, -3, 4, 5] * lambdaR4[4, -4] * lambdaR5[5, -5];
    

    # prepare and SVD ΓL
    ΓL = permutedims(ΓL, (5, 1, 2, 3, 4));
    sizeΓL = size(ΓL);
    UL, SL, VL = svd(reshape(ΓL, prod(sizeΓL[1 : 3]), prod(sizeΓL[4 : 5])));
    VL = VL';

    newChi = sum(SL .> 1e-12);
    UL = UL[:, 1 : newChi];
    SL = diagm(SL[1 : newChi]);
    VL = VL[1 : newChi, :];
    UL = permutedims(reshape(UL, sizeΓL[1 : 3]..., newChi), (2, 3, 4, 1));
    VL = permutedims(reshape(SL * VL, newChi, sizeΓL[4 : 5]...), (1, 2, 3));


    # prepare and SVD ΓR
    ΓR = permutedims(ΓR, (1, 3, 2, 4, 5));
    sizeΓR = size(ΓR);
    UR, SR, VR = svd(reshape(ΓR, prod(sizeΓR[1 : 2]), prod(sizeΓR[3 : 5])));
    VR = VR';

    newChi = sum(SR .> 1e-12);
    UR = UR[:, 1 : newChi];
    SR = diagm(SR[1 : newChi]);
    VR = VR[1 : newChi, :];
    UR = permutedims(reshape(UR * SR, sizeΓR[1 : 2]..., newChi), (1, 2, 3));
    VR = permutedims(reshape(VR, newChi, sizeΓR[3 : 5]...), (1, 2, 3, 4));


    # apply gate to reduced tensors
    @ein theta[-1, -2, -3, -4] := VL[-1, 3, 1] * lambdaLR[1, 2] * UR[2, 4, -4] * twoBodyGate[-2, -3, 3, 4];
    theta /= norm(theta);
    
    # SVD and truncate theta
    sizeTheta = size(theta);
    U, S, V = svd(reshape(theta, prod(sizeTheta[1 : 2]), prod(sizeTheta[3 : 4])));
    V = V';

    newChi = min(indBondDim[lambdaIdx], sum(S .> 1e-12))
    U = U[:, 1 : newChi];
    S = diagm(S[1 : newChi]);
    V = V[1 : newChi, :];
    U = reshape(U, sizeTheta[1 : 2]..., newChi);
    V = reshape(V, newChi, sizeTheta[3 : 4]...);
    S /= sqrt(tr(S^2));
    

    # update and regauge PEPS tensors
    @ein ΓL[-1, -2, -3, -4, -5] := UL[-1, -2, 1, -5] * U[1, -3, -4];
    @ein ΓR[-1, -2, -3, -4, -5] := V[-1, -3, 1] * VR[1, -2, -4, -5];
    @ein ΓL[-1, -2, -3, -4, -5] := pinv(lambdaL1)[-1, 1] * pinv(lambdaL2)[-2, 2] * ΓL[1, 2, -3, -4, 5] * pinv(lambdaL5)[5, -5];
    @ein ΓR[-1, -2, -3, -4, -5] := pinv(lambdaR2)[-2, 2] * ΓR[-1, 2, -3, 4, 5] * pinv(lambdaR4)[4, -4] * pinv(lambdaR5)[5, -5];

    # normalize ΓL and ΓR
    ΓL /= norm(ΓL);
    ΓR /= norm(ΓR);
    
    # store tensors
    gammaTensors[posΓL[1], posΓL[2]] = ΓL;
    gammaTensors[posΓR[1], posΓR[2]] = ΓR;
    lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2], 5]] = S;

    # function return
    return gammaTensors, lambdaTensors;

end

# simple update on vertical bonds
function simpleUpdate_iPEPS_V(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate)

    # determine gamma and lambda tensors
    posΓL = [lambdaIndexTable[updateLambdaMatrix[2],1], lambdaIndexTable[updateLambdaMatrix[2],2]];
    posΓR = [lambdaIndexTable[updateLambdaMatrix[3],1], lambdaIndexTable[updateLambdaMatrix[3],2]];
    ΓL = gammaTensors[posΓL[1], posΓL[2]];
    ΓR = gammaTensors[posΓR[1], posΓR[2]];
    
    lambdaL1 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],3]];
    lambdaL2 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],4]];
    lambdaL4 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],5]];

    lambdaLR = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2],6]];

    lambdaR1 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],3]];
    lambdaR4 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],5]];
    lambdaR5 = lambdaTensors[lambdaIndexTable[updateLambdaMatrix[3],6]];
    
    # absorb all lambdas into ΓL and ΓR
    @ein ΓL[-1, -2, -3, -4, -5] := lambdaL1[-1, 1] * lambdaL2[-2, 2] * ΓL[1, 2, -3, 4, -5] * lambdaL4[4, -4];
    @ein ΓR[-1, -2, -3, -4, -5] := lambdaR1[-1, 1] * ΓR[1, -2, -3, 4, 5] * lambdaR4[4, -4] * lambdaR5[5, -5];

    # prepare and SVD ΓL
    ΓL = permutedims(ΓL, (1, 2, 4, 3, 5));
    sizeΓL = size(ΓL);
    UL, SL, VL = svd(reshape(ΓL, prod(sizeΓL[1 : 3]), prod(sizeΓL[4 : 5])));
    VL = VL';

    newChi = sum(SL .> 1e-12);
    UL = UL[:, 1 : newChi];
    SL = diagm(SL[1 : newChi]);
    VL = VL[1 : newChi, :];
    UL = permutedims(reshape(UL, sizeΓL[1 : 3]..., newChi), (1, 2, 3, 4));
    VL = permutedims(reshape(SL * VL, newChi, sizeΓL[4 : 5]...), (1, 2, 3));
    

    # prepare and SVD ΓR
    ΓR = permutedims(ΓR, (2, 3, 4, 5, 1));
    sizeΓR = size(ΓR);
    UR, SR, VR = svd(reshape(ΓR, prod(sizeΓR[1 : 2]), prod(sizeΓR[3 : 5])));
    VR = VR';

    newChi = sum(SR .> 1e-12);
    UR = UR[:, 1 : newChi];
    SR = diagm(SR[1 : newChi]);
    VR = VR[1 : newChi, :];
    UR = permutedims(reshape(UR * SR, sizeΓR[1 : 2]..., newChi), (1, 2, 3));
    VR = permutedims(reshape(VR, newChi, sizeΓR[3 : 5]...), (4, 1, 2, 3));
    

    # apply gate to reduced tensors
    @ein theta[-1, -2, -3, -4] := VL[-1, 3, 1] * lambdaLR[1, 2] * UR[2, 4, -4] * twoBodyGate[-2, -3, 3, 4];
    theta /= norm(theta);
    
    # SVD and truncate theta
    sizeTheta = size(theta);
    U, S, V = svd(reshape(theta, prod(sizeTheta[1 : 2]), prod(sizeTheta[3 : 4])));
    V = V';

    newChi = min(indBondDim[lambdaIdx], sum(S .> 1e-12));
    U = U[:, 1 : newChi];
    S = diagm(S[1 : newChi]);
    V = V[1 : newChi, :];
    U = reshape(U, sizeTheta[1 : 2]..., newChi);
    V = reshape(V, newChi, sizeTheta[3 : 4]...);
    S /= sqrt(tr(S^2));
    

    # update and regauge PEPS tensors
    @ein ΓL[-1, -2, -3, -4, -5] := UL[-1, -2, -4, 1] * U[1, -3, -5];
    @ein ΓR[-1, -2, -3, -4, -5] := V[-2, -3, 1] * VR[-1, 1, -4, -5];
    @ein ΓL[-1, -2, -3, -4, -5] := pinv(lambdaL1)[-1, 1] * pinv(lambdaL2)[-2, 2] * ΓL[1, 2, -3, 4, -5] * pinv(lambdaL4)[4, -4];
    @ein ΓR[-1, -2, -3, -4, -5] := pinv(lambdaR1)[-1, 1] * ΓR[1, -2, -3, 4, 5] * pinv(lambdaR4)[4, -4] * pinv(lambdaR5)[5, -5];
    
    # normalize ΓL and ΓR
    ΓL /= norm(ΓL);
    ΓR /= norm(ΓR);

    # store PEPS tensors and lambda in arrays
    gammaTensors[posΓL[1], posΓL[2]] = ΓL;
    gammaTensors[posΓR[1], posΓR[2]] = ΓR;
    lambdaTensors[lambdaIndexTable[updateLambdaMatrix[2], 6]] = S;

    # function return
    return gammaTensors, lambdaTensors;

end

# function to print simple update convergence information
function verbosePrintSimpleUpdate(loopCounter::Int64, controlTimeStep::Int64, normSingularValue::Float64)
    @info("Simple Update", loopCounter, controlTimeStep, normSingularValue)
end

# simple update
function simpleUpdate_iPEPS(Lx, Ly, d, chiB, convTolB, energyTBG; verbosePrint = false)

    # determine unitCell, lambdaIndexTable and lambdaUpdateTensors for square lattice simple update
    numLinks, lambdaIndexTable, lambdaUpdateTensors = arbitraryUnitCell_SquareLattice(Lx, Ly);
    # println(lambdaIndexTable)
    # println(lambdaUpdateTensors)

    # set individual bond dimensions
    iniBondDim = ones(Int64, numLinks);
    indBondDim = chiB * ones(Int64, numLinks);
    # indBondDim = [3, 4, 5, 6];

    # initialize iPEPS unit cell
    gammaTensors, lambdaTensors = initialize_iPEPS_UnitCell(Lx, Ly, iniBondDim, d, numLinks, lambdaIndexTable);

    # maximal number of steps
    maxNumSteps = Integer(1e6);
    minIterationsPerGate = 50;
    stepDistanceCheck = 30;

    verboseInd = false;

    # reshape energyTBG to matrix
    sizeTBG = size(energyTBG);
    energyTBG = reshape(energyTBG, prod(sizeTBG[1 : 2]), prod(sizeTBG[3 : 4]))

    # construct Suzuki-Trotter gates
    timeSteps = [1e-1, 1e-2, 1e-3];
    gatesTimeSteps = Vector{Array{Float64, 4}}(undef, length(timeSteps));
    for gateIdx = 1 : length(timeSteps)
        gatesTimeSteps[gateIdx] = permutedims(reshape(exp(-timeSteps[gateIdx] * energyTBG), sizeTBG), (1, 2, 3, 4));
    end

    # get number of different Suzuki-Trotter steps
    maxControlTimeStep = length(gatesTimeSteps);

    # initialize list for singular values
    singularValueTensor = zeros(Float64, maxNumSteps, 1 + maximum(indBondDim), numLinks);
    for lambdaIdx = 1 : numLinks
        singularValueTensor[:, 1, lambdaIdx] = 1 : maxNumSteps;
    end

    # print simple update info
    @printf("\nRunning Simple Update...\n")

    # simple update parameter
    loopCounter = 1;
    gateCounter = 1;
    controlTimeStep = 1;
    normSingularValue = 1.0;
    runSimulation = 1;
    while runSimulation == 1 && loopCounter <= maxNumSteps

        # select which two-body gate to use
        twoBodyGate = gatesTimeSteps[controlTimeStep];

        # loop over all links in the iPEPS network
        for lambdaIdx = 1 : numLinks
            
            # select order of PEPS and lambda tensors for update of specific lambda
            updateLambdaMatrix = lambdaUpdateTensors[lambdaIdx,:];
            
            if isequal(updateLambdaMatrix[4 : 5], [3, 1])
                verboseInd && println("updating X-link...\n")
                updateLambdaMatrix = updateLambdaMatrix[[1, 2, 3, 4, 5]];
                gammaTensors, lambdaTensors = simpleUpdate_iPEPS_H(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate);
            elseif isequal(updateLambdaMatrix[4 : 5], [1, 3])
                verboseInd && println("updating Y-link...\n")
                updateLambdaMatrix = updateLambdaMatrix[[1, 3, 2, 5, 4]];
                gammaTensors, lambdaTensors = simpleUpdate_iPEPS_H(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate);
            elseif isequal(updateLambdaMatrix[4 : 5], [2, 4])
                verboseInd && println("updating Z-link...\n")
                updateLambdaMatrix = updateLambdaMatrix[[1, 3, 2, 5, 4]];
                gammaTensors, lambdaTensors = simpleUpdate_iPEPS_V(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate);
            elseif isequal(updateLambdaMatrix[4 : 5], [4, 2])
                verboseInd && println("updating D-link...\n")
                updateLambdaMatrix = updateLambdaMatrix[[1, 2, 3, 4, 5]];
                gammaTensors, lambdaTensors = simpleUpdate_iPEPS_V(lambdaIdx, lambdaIndexTable, updateLambdaMatrix, gammaTensors, lambdaTensors, indBondDim, twoBodyGate);
            end
            
        end


        #-----------------------------------------------------------------------
        # store singular values and determine convergence
        #-----------------------------------------------------------------------

        # store singular values
        for lambdaIdx = 1 : numLinks
            singularValues = svd(lambdaTensors[lambdaIdx]).S;
            singularValueTensor[loopCounter,1 .+ (1 : length(singularValues)),lambdaIdx] = singularValues;
        end

        # check convergence of singular values
        if loopCounter > stepDistanceCheck

            normSingularValue = norm(reshape(singularValueTensor[loopCounter, 2 : end, :], maximum(indBondDim), numLinks) .- reshape(singularValueTensor[loopCounter - stepDistanceCheck, 2 : end,:], maximum(indBondDim), numLinks));
            if (normSingularValue < convTolB) && (gateCounter >= minIterationsPerGate)
                if controlTimeStep < maxControlTimeStep
                    controlTimeStep += 1;
                    gateCounter = 1;
                else
                    runSimulation = 0;
                end
            end

        end

        # print convergence information
        verbosePrint && verbosePrintSimpleUpdate(loopCounter, controlTimeStep, normSingularValue)

        # increase loopCounter
        loopCounter += 1;
        gateCounter += 1;

    end

    loopCounter -= 1;
    singularValueTensor = singularValueTensor[1 : loopCounter, :, :];

    @printf("Simple Update Completed with %d Steps, normSingularValues %0.8e\n", loopCounter, normSingularValue)

    # absorb lambdas matrices back into iPEPS tensors
    for gammaIdx = 1 : size(lambdaIndexTable,1)
        
        # get relevant PEPS tensor and lambda matrices
        posΓX = lambdaIndexTable[gammaIdx,1];
        posΓY = lambdaIndexTable[gammaIdx,2];
        gammaT = gammaTensors[posΓX, posΓY];
        lambdaT1 = lambdaTensors[lambdaIndexTable[gammaIdx,3]];
        lambdaT2 = lambdaTensors[lambdaIndexTable[gammaIdx,4]];
        lambdaT4 = lambdaTensors[lambdaIndexTable[gammaIdx,5]];
        lambdaT5 = lambdaTensors[lambdaIndexTable[gammaIdx,6]];
        
        # absorb lambda tensors and store gammaT
        @ein gammaT[-1, -2, -3, -4, -5] := sqrt(lambdaT1)[-1, 1] * sqrt(lambdaT2)[-2, 2] * gammaT[1, 2, -3, 4, 5] * sqrt(lambdaT4)[4, -4] * sqrt(lambdaT5)[5, -5];
        gammaTensors[posΓX, posΓY] = gammaT;
        
    end

    # return updated iPESS unit cell
    return gammaTensors, lambdaTensors, singularValueTensor;

end