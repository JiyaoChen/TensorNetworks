using TensorKit

function generateHeisenbergNoSym(P::Dict)
    
    # set vector spaces
    vP = ℂ^2
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