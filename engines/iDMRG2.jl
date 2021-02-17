# additional contractions used in this scope
include("iDMRG2_contractions.jl")

# traditional growing algorithm -- starts from scratch with a given mpo tensor and variationally searches for the (2-site periodic) MPS
function iDMRG2(mpo::A; χ::Int64=64, steps::Int64=100, tol::Float64=KrylovDefaults.tol) where {A<:AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}}
    # this extracts the link spaces from the MPO tensor legs
    vP = space(mpo)[1]
    vL = space(mpo)[2]
    # this is an outgoing leg so it must be conjugated for further purposes
    vR = conj(space(mpo)[3])
    # initial legs of the MPS (currently only ℤ₂ and ℂ, needs further adaption)
    if typeof(vP) <: GradedSpace
        vV = typeof(space(mpo)[1])(0=>1)
    elseif typeof(vP) <: CartesianSpace
        vV = typeof(space(mpo)[1])(1)
    end

    # initialize MPS tensors
    gammaTensor_arr = zeros(ComplexF64, dim(vP), dim(vV), dim(vV));
    gammaTensor_arr[1,1,1] = 1;
    gammaTensor = TensorMap(gammaTensor_arr, vP*vV, vV);
    gammaList = fill(gammaTensor,2);

    lambdaTensor_arr = zeros(ComplexF64, dim(vV), dim(vV));
    lambdaTensor_arr[1,1] = 1;
    lambdaTensor = TensorMap(lambdaTensor_arr, vV, vV);
    lambdaList = fill(lambdaTensor,2);

    # initiliaze EL and ER
    EL_arr = zeros(ComplexF64,dim(vV),dim(vL),dim(vV));
    EL_arr[1,1,1] = 1;
    EL = TensorMap(EL_arr, vV, vL*vV);

    ER_arr = zeros(ComplexF64,dim(vR),dim(vV),dim(vV));
    ER_arr[2,1,1] = 1;
    ER = TensorMap(ER_arr, vR*vV, vV);

    groundStateEnergy = zeros(Float64, steps, 5);

    # initialize variables to be available outside of for-loop
    ϵ = 0
    current_χ = 0
    currEigenVal = 0;
    currEigenVec = [];
    prevEigenVal = 0;
    prevEigenVec = [];

    for i = 1 : steps
        
        groundStateEnergy[i,1] = i;

        for i_bond = 1 : 2
            # choose unit cell arrangement
            ia = mod(i_bond + 0,2) + 1;
            ib = mod(i_bond + 1,2) + 1;
            ic = mod(i_bond + 2,2) + 1;

            # construct initial wave function
            theta = initialWF(lambdaList[ia], gammaList[ib], gammaList[ic]);
            
            # store previous eivenvalue
            prevEigenVal = currEigenVal;
            prevEigenVec = theta;
            
            # optimize wave function
            eigenVal, eigenVec =
                eigsolve(theta,1, :SR, Arnoldi(tol=tol)) do x
                    applyH(x, EL, mpo, ER)
                end
            # eigenVal, eigenVec = eigsolve(x -> applyH(x, EL, mpo, ER), theta, 1, :SR, Lanczos());
            currEigenVal = eigenVal[1];
            currEigenVec = eigenVec[1];
            
            #  perform SVD and truncate to desired bond dimension
            U, S, V, ϵ = tsvd(currEigenVec, (2,3), (1,4), trunc = truncdim(χ));
            # print(S);

            current_χ = dim(space(S,1));
            V = permute(V, (2,1), (3,));

            # update environments
            EL = update_EL(EL, U, mpo);
            ER = update_ER(ER, V, mpo);

            # obtain the new tensors for MPS
            lambdaList[ib] = S;
            gammaList[ib] = new_gamma(lambdaList[ia], U, lambdaList[ib]);
            gammaList[ic] = V;

        end

        # calculate ground state energy
        gsEnergy = 1/2*(currEigenVal - prevEigenVal);

        # # calculate overlap between old and new wave function
        # @tensor waveFuncOverlap[:] := currEigenVec[1 2 3] * conj(prevEigenVec[1 2 3]);

        # print simulation progress
        @printf("%05i : E_iDMRG / Convergence / Discarded Weight / BondDim : %0.15f / %0.15f / %d \n",i,real(gsEnergy),ϵ,current_χ)
    end

    return gammaList
end