using TensorKit

function generateHeisenbergSU2(P::Dict)
    
    # set vector spaces
    spinS = 1/2
    vP = SU₂Space(spinS => 1)
    vM = SU₂Space(0 => 2, 1 => 1)

    γ = sqrt(spinS * (spinS + 1))

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
        dataDict["Irrep[SU₂](1/2)"] = Array{ComplexF64}(ham_arr)
        tensorDict[:data] = dataDict
        
        mpos[idx] = convert(TensorMap, tensorDict)

    end

    return DMRG_types.MPO([mpo for mpo in mpos])

end