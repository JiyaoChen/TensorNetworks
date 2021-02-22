# additional contractions used in this scope
include("iDMRG2_contractions.jl")
include("iDMRG2_link_manipulations.jl")
using DMRG_types

# traditional growing algorithm -- starts from scratch with a given mpo tensor and variationally searches for the (2-site periodic) MPS
function iDMRG2(mpo::A; χ::Int64=64, numSteps::Int64=100, tol::Float64=KrylovDefaults.tol) where {A<:AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}}

    # mpo_arr = convert(Array, mpo)
    # mpo = TensorMap(mpo_arr, ComplexSpace(2)*ComplexSpace(5), ComplexSpace(5)*ComplexSpace(2))
    
    # this extracts the link spaces from the MPO tensor legs
    physSpace = space(mpo)[2]
    mpoSpaceL = space(mpo)[3]'  # this is an incoming leg so it must be conjugated
    mpoSpaceR = space(mpo)[1]

    # initial legs of the MPS (currently only ℤ₂ and ℂ, needs further adaption)
    if occursin("ComplexSpace",string(typeof(physSpace)))
        zeroIrrep = ℂ^1
    elseif occursin("ZNIrrep{2}",string(typeof(physSpace)))
        zeroIrrep = ℤ₂Space(0 => 1)
        oneIrrep = ℤ₂Space(0 => 1)
    elseif occursin("U1Irrep",string(typeof(physSpace)))
        zeroIrrep = U1Space(0 => 1)
        oneIrrep = U1Space(1 => 1)
    elseif  occursin("SU2Irrep",string(typeof(physSpace)))
        zeroIrrep = SU₂Space(0 => 1)
        oneIrrep = SU₂Space(0 => 1)
    end
    mpsSpaceL = zeroIrrep
    mpsSpaceR = oneIrrep
    mpoSpaceI = zeroIrrep
    mpoSpaceO = zeroIrrep
    mpsSpaceShared = computeSharedLink(mpsSpaceL, physSpace, physSpace, mpsSpaceR)
    
    mpoBoundaryVecL = zeros(ComplexF64, dim(mpoSpaceL), dim(mpoSpaceI))
    mpoBoundaryVecL[1] = 1
    # mpoBoundaryVecL = Array{ComplexF64}([1.0 0.0]);
    mpoBoundaryVecR = zeros(ComplexF64, dim(mpoSpaceO), dim(mpoSpaceR))
    mpoBoundaryVecR[2] = 1
    # mpoBoundaryVecR = Array{ComplexF64}([0.0 ; 1.0]);
    mpoBoundaryTensL = TensorMap(mpoBoundaryVecL, mpoSpaceI, mpoSpaceL)
    mpoBoundaryTensR = TensorMap(mpoBoundaryVecR, mpoSpaceR, mpoSpaceO)

    # mpoBoundaryTensL = TensorMap(reshape([1 0], (1 1 2)), mpoSpaceI*mpoSpaceL, zeroIrrep)
    # mpoBoundaryTensL = TensorMap(zeros, mpoSpaceI, mpoSpaceL)
    # tensorDictL = convert(Dict, mpoBoundaryTensL)
    # dataDictL = tensorDictL[:data]
    # dataDictL["Irrep[SU₂](0)"] = Array{ComplexF64}([1.0 0.0])
    # tensorDictL[:data] = dataDictL
    # mpoBoundaryTensL = convert(TensorMap, tensorDictL)

    # mpoBoundaryTensR = TensorMap(zeros, mpoSpaceR, mpoSpaceO)
    # tensorDictR = convert(Dict, mpoBoundaryTensR)
    # dataDictR = tensorDictR[:data]
    # dataDictR["Irrep[SU₂](0)"] = Array{ComplexF64}(reshape([0.0 ; 1.0], (2,1)))
    # tensorDictR[:data] = dataDictR
    # mpoBoundaryTensR = convert(TensorMap, tensorDictR)
    # mpoBoundaryTensL = reshape(mpoBoundaryTensL, dim())
    # mpoBoundaryTensL = TensorMap(mpoBoundaryTensL, space(mpoBoundaryTensL, 1), space(mpoBoundaryTensL, 2))
    # mpoBoundaryTensR = Tensor([0 1], mpoSpaceR*mpoSpaceO)

    # println(mpoBoundaryTensL)

    # initialize MPS tensors
    T1 = TensorMap(randn, ComplexF64, mpsSpaceL ⊗ physSpace, mpsSpaceShared)
    T2 = TensorMap(randn, ComplexF64, mpsSpaceShared ⊗ physSpace, mpsSpaceR)

    # U, Spr, Vdag, ϵ = tsvd(T1*permute(T2, (1,), (2,3)), (1,2), (3,4), trunc = truncdim(χ))
    # print(T1)
    # println(U)
    # print(T2)
    # print(permute(Vdag, (1,2), (3,)))

    # initiliaze EL and ER
    IdL = TensorMap(ones, ComplexF64, mpsSpaceL, mpoSpaceI ⊗ mpsSpaceL)
    IdR = TensorMap(ones, ComplexF64, mpsSpaceR ⊗ mpoSpaceO, mpsSpaceR)
    @tensor EL[-1; -2 -3] := IdL[-1 1 -3] * mpoBoundaryTensL[1 -2]
    @tensor ER[-1 -2; -3] := mpoBoundaryTensR[-2 1] * IdR[-1 1 -3]

    
    # return
    
    # initialize array to store energy
    groundStateEnergy = zeros(Float64, numSteps, 5)

    # initialize variables to be available outside of for-loop
    ϵ = 0
    current_χ = 0
    currEigenVal = 0
    currEigenVec = []
    prevEigenVal = 0
    prevEigenVec = []

    # initialize tensorTrain
    # tensorTrain = Vector{A}(undef,2*numSteps) where {T<:Number, S<:Array{T}}
    # tensorTrain = Vector{A}(undef,2*numSteps) where {A<:AbstractTensorMap}
    # tensorTrain = Vector{Any}(undef,4*numSteps)
    # tensorTrain = {}

    
    # construct initial wave function
    theta = permute(T1 * permute(T2, (1,), (2,3)), (1,2,3), (4,))
    Spr = TensorMap(ones, zeroIrrep, zeroIrrep);
    
    # main growing loop
    for i = 1 : numSteps
        
        groundStateEnergy[i,1] = i
        
        # store previous eivenvalue
        prevEigenVal = currEigenVal;
        
        # optimize wave function
        eigenVal, eigenVec = 
            eigsolve(theta,1, :SR, Arnoldi(tol=tol)) do x
                applyH(x, EL, mpo, ER)
            end
        # println(eigenVal)
        currEigenVal = eigenVal[1]
        currEigenVec = eigenVec[1]
        
        #  perform SVD and truncate to desired bond dimension
        S = Spr
        U, Spr, Vdag, ϵ = tsvd(currEigenVec, (1,2), (3,4), trunc = truncdim(χ))

        current_χ = dim(space(Spr,1))
        Vdag = permute(Vdag, (1,2), (3,))

        # update environments
        EL = update_EL(EL, U, mpo)
        ER = update_ER(ER, Vdag, mpo)

        # shift quantum numbers
        ER = shiftVirtSpaceMPS(ER, oneIrrep)

        # save the tensors
        # tensorTrain[2*(i-1)+i_bond] = U
        # tensorTrain[end-2*(i-1)-i_bond+1] = V

        # obtain the new tensors for MPS
        theta = newGuess(oneIrrep, Spr, Vdag, S, U)
        # theta_shifted = TensorMap(convert(Array, theta), space(EL)[3]' ⊗ space(theta)[2] ⊗ space(theta)[3], space(ER)[1]')
        # print(theta_shifted)

        # calculate ground state energy
        gsEnergy = 1/2*(currEigenVal - prevEigenVal)

        # # calculate overlap between old and new wave function
        # @tensor waveFuncOverlap[:] := currEigenVec[1 2 3] * conj(prevEigenVec[1 2 3]);

        # print simulation progress
        @printf("%05i : E_iDMRG / Convergence / Discarded Weight / BondDim : %0.15f / %0.15f / %d \n",i,real(gsEnergy),ϵ,current_χ)
        
    end
    
    # tensorTrain[2*numSteps+1] = gammaList[1]
    # tensorTrain[2*numSteps] = gammaList[2]

    # mps = DMRG_types.MPS([tensor for tensor in tensorTrain]);

    print(space(Spr),"\n")

    return
    
end