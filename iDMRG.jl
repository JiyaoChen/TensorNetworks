
function iDMRG(bondDim::Int64,numSteps::Int64)

    J = 1;
    h = 0;

    chi = bondDim;
    maxNumSteps = numSteps;
    d = 2;

    # set convergence parameters
    eigsTol = 1e-15;
    convTolE = 1e-12;
    convTolW = 1e-14;

    # Pauli opertators
    Id = Matrix{ComplexF64}(I,2,2);
    X = 1/2 * [ 0 +1 ; +1 0 ];
    Y = 1/2 * [ 0 -1im ; +1im 0 ];
    Z = 1/2 * [ +1 0 ; 0 -1 ];

    # generate the Heisenberg MPO
    H_MPO = zeros(Complex{Float64},5,5,2,2);
    H_MPO[1,1,:,:] = Id;
    H_MPO[1,2,:,:] = X;
    H_MPO[1,3,:,:] = Y;
    H_MPO[1,4,:,:] = Z;
    H_MPO[1,5,:,:] = -h*Z;
    H_MPO[2,5,:,:] = J*X;
    H_MPO[3,5,:,:] = J*Y;
    H_MPO[4,5,:,:] = J*Z;
    H_MPO[5,5,:,:] = Id;
    dMPO = size(H_MPO,1);

    # generate gammaList
    gammaList = Array{Array{Float64,3}}(undef,1,2);
    for i = 1 : size(gammaList,2)
        gammaList[i] = zeros(d,1,1);
        gammaList[i][1,1,1] = 1;
    end

    # generate lambdaList
    lambdaList = Array{Array{Float64,2}}(undef,1,2)
    for i = 1 : size(lambdaList,2)
        lambdaList[i] = ones(Float64,1,1);
    end

    # combine the lists to one MPS
    MPS = [gammaList[1],lambdaList[1],gammaList[2],lambdaList[2]];

    # initiliaze left and right environment
    EL = zeros(Float64,1,1,dMPO);
    EL[1,1,1] = 1;
    ER = zeros(Float64,1,1,dMPO);
    ER[1,1,dMPO] = 1;

    numElementsGroundStateArray = 5;
    groundStateEnergy = zeros(Float64,maxNumSteps,numElementsGroundStateArray);

    # initialize variables to be available outside of for-loop
    eigenVal = 0;
    prevEigenVal = 0;
    eigenVec = [];
    prevEigenVec = [];
    discardedWeight = 0;

    for i = 1 : maxNumSteps

        groundStateEnergy[i,1] = i;

        for i_bond = 1 : 2

            # choose unit cell arrangement
            ia = mod(i_bond + 0,2) + 1;
            ib = mod(i_bond + 1,2) + 1;
            ic = mod(i_bond + 2,2) + 1;

            # get dimensions of tensors to optimize
            chia = size(gammaList[ib],2);
            chic = size(gammaList[ic],3);

            # construct initial wave function
            @tensor thetaInit[a,b,c,d] := lambdaList[ia][a,e] * gammaList[ib][b,e,f] * gammaList[ic][c,f,d];
            thetaInit = reshape(thetaInit,(chia*d*d*chic));
            thetaInit = thetaInit ./ sqrt(abs(dot(thetaInit,thetaInit)));

            # store previous eivenvalue
            prevEigenVal = eigenVal;
            prevEigenVec = thetaInit;

            # function to apply the Hamiltonian to the wave function
            function effectiveHamiltonian_twoSite(src::AbstractVector)

                # reshape vector to tensor
                src = reshape(src,(chia,d,d,chic));

                # perform two-site update
                @tensor dest[:] := EL[1,-1,2] * src[1,3,5,7] * H_MPO[2,4,3,-2] * H_MPO[4,6,5,-3] * ER[7,-4,6];

                # return new wavefunction
                return dest

            end

            # optimize wave function
            D = LinearMap(effectiveHamiltonian_twoSite,chia*d*chic*d);
            eigenVal , eigenVec = eigs(D,nev = 1,which = :SR,tol = 13-13,v0 = thetaInit);

            # reshape waveFunc to matrix and perform SVD
            waveFunc = reshape(eigenVec,(chia*d,d*chic));
            U,S,V = svd(waveFunc);
            V = V';

            # truncation of the singular values
    		chib = min(sum(S .> 1e-15),chi);
    		U = permutedims(reshape(U[1 : size(U,1),1 : chib],(chia,d,chib)),(2,1,3));
            S = diagm(S);
            S = S[1 : chib,1 : chib];
            discardedWeight = abs(1 - tr(S.^2));
            V = permutedims(reshape(V[1 : chib,1 : size(V,2)],(chib,d,chic)),(2,1,3));

            # update environment
            @tensor EL[:] := EL[1,4,2] * U[3,1,-1] * H_MPO[2,-3,3,5] * conj(U)[5,4,-2];
            @tensor ER[:] := ER[1,4,2] * V[3,-1,1] * H_MPO[-3,2,3,5] * conj(V)[5,-2,4];

            # obtain the new tensors for MPS
            lambdaList[ib] = S ./ sqrt(sum(diag(S.^2)));
            @tensor gammaList[ib][:] := inv(lambdaList[ia])[-2,1] * U[-1,1,2] * lambdaList[ib][2,-3];
            gammaList[ic] = V;

        end

        # calculate ground state energy
        gsenergy = 1/2*(eigenVal - prevEigenVal);
        gsenergy = gsenergy[1];

        # calculate overlap between old and new wave function
        overlapTwoSite = prevEigenVec' * eigenVec;
        overlapTwoSite = overlapTwoSite[1];

        groundStateEnergy[i,2] = real(gsenergy);
        groundStateEnergy[i,3] = imag(gsenergy);
        groundStateEnergy[i,4] = overlapTwoSite;
        groundStateEnergy[i,5] = discardedWeight;

        # print simulation progress
        @printf("%05i : E_iDMRG / Convergence / Discarded Weight : %0.15f / %0.15f / %0.15f\n",i,real(gsenergy),real(overlapTwoSite[1]),discardedWeight)

    end

    MPS = [gammaList[1],lambdaList[1],gammaList[2],lambdaList[2]];

    return MPS , groundStateEnergy

end
