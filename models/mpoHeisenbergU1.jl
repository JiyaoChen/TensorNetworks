using TensorKit

function generateHeisenbergU1(P::Dict)
    # set vector spaces
    vP = U1Space(0 => 1, 1 => 1)
    vM = U1Space(0 => 3, -1 => 1, +1 => 1)

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    # fill in MPOs
    mpos = Vector{Any}(undef,P["length"])
    Js = fill(1.0, length(mpos))
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