using TensorKit
include("getSpinOperators.jl")

function mpoHeisenberg(; J::Float64 = 1.0, spinS::Float64 = 1/2, setSym = "")

    # construct MPOs with different symmetries
    if setSym == ""

        # set vector spaces
        d = Int(2 * spinS + 1);
        vP = ℂ^d;
        vM = ℂ^5;

        # get spin operators
        Sx, Sy, Sz, Id = getSpinOperators(spinS);

        # generate the Heisenberg MPO
        ham_arr = zeros(ComplexF64, dim(vM), dim(vP), dim(vM), dim(vP));
        ham_arr[1,:,1,:] = Id;
        ham_arr[2,:,2,:] = Id;
        ham_arr[1,:,3,:] = J * Sx;
        ham_arr[1,:,4,:] = J * Sy;
        ham_arr[1,:,5,:] = J * Sz;
        ham_arr[3,:,2,:] = Sx;
        ham_arr[4,:,2,:] = Sy;
        ham_arr[5,:,2,:] = Sz;
        
        mpo = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)

    elseif setSym == "U1"

        # set vector spaces
        vP = U1Space(0 => 1, 1 => 1)
        vM = U1Space(0 => 3, -1 => 1, +1 => 1)

        # get spin operators
        Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(spinS)
        ham_arr = zeros(ComplexF64, dim(vM), dim(vP), dim(vM), dim(vP))
        ham_arr[1,:,1,:] = Id
        ham_arr[2,:,2,:] = Id
        ham_arr[1,:,4,:] = 0.5 * J * Sm
        ham_arr[1,:,5,:] = 0.5 * J * Sp
        ham_arr[1,:,3,:] = J * Sz
        ham_arr[4,:,2,:] = Sp
        ham_arr[5,:,2,:] = Sm
        ham_arr[3,:,2,:] = Sz

        mpo = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)
    
    elseif setSym == "SU2"

        # TODO this does not work with the VSCode debugger...

        # set vector spaces 
        vP = SU₂Space(spinS => 1)
        mpoSpaceL = SU₂Space(0 => 2, 1 => 1)
        mpoSpaceR = SU₂Space(0 => 2, 1 => 1)
        γ = sqrt(spinS * (spinS + 1))

        # construct empty MPO
        mpo = TensorMap(zeros, ComplexF64, mpoSpaceL ⊗ vP, mpoSpaceR ⊗ vP)

        # fill tensor blocks
        tensorDict = convert(Dict, mpo)
        dictKeys = keys(tensorDict)
        dataDict = tensorDict[:data]
        dataDict["Irrep[SU₂](1/2)"] = Array{ComplexF64}([1.0 0.0 J * γ ; 0.0 1.0 0.0 ; 0.0 γ 0.0])
        tensorDict[:data] = dataDict
        mpo = convert(TensorMap, tensorDict)

    end

    # function return
    return mpo

end

function generateHeisenbergNoSym(P::Dict)
    
    # set vector spaces
    vP = ℂ^Int64(floor(real(2*P["spin"]+1)))
    vM = ℂ^5

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    # fill in MPOs
    mpos = Vector{Any}(undef,P["length"])
    Js = fill(1.0, length(mpos))
    # this does not at all make sense for huge systems -- better store objects in MPO and overload the getters
    for (idx, J) in enumerate(Js)

        ham_arr = zeros(P["eltype"], dim(vM), dim(vP), dim(vM), dim(vP))

        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx != length(Js)
            ham_arr[1,:,1,:] = Id
            ham_arr[1,:,4,:] = 0.5 * J * Sm
            ham_arr[1,:,5,:] = 0.5 * J * Sp
            ham_arr[1,:,3,:] = J * Sz
        end
        if idx != 1
            ham_arr[4,:,2,:] = Sp
            ham_arr[5,:,2,:] = Sm
            ham_arr[3,:,2,:] = Sz
            ham_arr[2,:,2,:] = Id
        end
        mpos[idx] = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)

    end

    return DMRG_types.MPO([mpo for mpo in mpos])

end

function generateHeisenbergU1(P::Dict)
    
    # set vector spaces
    vM = U1Space(0 => 3, -1 => 1, +1 => 1)

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    numSectors = size(Sz, 1)
    vP = U1Space([numSectors-i => 1 for i = 1 : numSectors])

    # fill in MPOs
    mpos = Vector{Any}(undef,P["length"])
    Js = fill(1.0, length(mpos))
    # this does not at all make sense for huge systems -- better store objects in MPO and overload the getters
    for (idx, J) in enumerate(Js)

        ham_arr = zeros(P["eltype"], dim(vM), dim(vP), dim(vM), dim(vP))

        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx != length(Js)
            ham_arr[1,:,1,:] = Id
            ham_arr[1,:,4,:] = 0.5 * J * Sm
            ham_arr[1,:,5,:] = 0.5 * J * Sp
            ham_arr[1,:,3,:] = J * Sz
        end
        if idx != 1
            ham_arr[4,:,2,:] = Sp
            ham_arr[5,:,2,:] = Sm
            ham_arr[3,:,2,:] = Sz
            ham_arr[2,:,2,:] = Id
        end
        mpos[idx] = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)

    end

    return DMRG_types.MPO([mpo for mpo in mpos])

end

function generateHeisenbergSU2(P::Dict)
    
    # set vector spaces
    vP = SU₂Space(P["spin"] => 1)
    vM = SU₂Space(0 => 2, 1 => 1)

    γ = sqrt(P["spin"] * (P["spin"] + 1))

    # fill in MPOs
    mpos = Vector{Any}(undef,P["length"])
    Js = fill(1.0, length(mpos))
    # this does not at all make sense for huge systems -- better store objects in MPO and overload the getters
    for (idx, J) in enumerate(Js)
        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx == 1
            ham_arr = [1.0 0.0 J * γ ; 0.0 0.0 0.0 ; 0.0 0 0.0]
        end
        if idx != length(Js) && idx != 1
            ham_arr = [1.0 0.0 J * γ ; 0.0 1.0 0.0 ; 0.0 γ 0.0]
        end
        if idx == length(Js)
            ham_arr = [0.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 γ 0.0]
        end
        
        # construct empty MPO
        mpo = TensorMap(zeros, ComplexF64, vM ⊗ vP, vM ⊗ vP)
        # fill tensor blocks
        tensorDict = convert(Dict, mpo)
        dictKeys = keys(tensorDict)
        dataDict = tensorDict[:data]
        irrepStr = "Irrep[SU₂](" * string(P["spin"]) * ")"
        dataDict[irrepStr] = Array{ComplexF64}(ham_arr)
        tensorDict[:data] = dataDict
        
        mpos[idx] = convert(TensorMap, tensorDict)

    end

    return DMRG_types.MPO([mpo for mpo in mpos])

end