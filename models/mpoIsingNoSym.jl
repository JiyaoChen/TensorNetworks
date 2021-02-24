using TensorKit

function generateIsingNoSym(P::Dict)
    
    # set vector spaces
    vP = ℂ^2
    vM = ℂ^3

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(1/2)

    elemtype = Float64

    # fill in MPOs
    mpos = Vector{Any}(undef,P["length"])
    Js = fill(4.0, length(mpos))
    hs = fill(2.0, length(Js))
    # this does not at all make sense for huge systems -- better store objects in MPO and overload the getters
    for (idx, J) in enumerate(Js)

        ham_arr = zeros(elemtype, dim(vM), dim(vP), dim(vM), dim(vP))

        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx != length(Js)
            ham_arr[1,:,1,:] = Id;
            ham_arr[1,:,3,:] = J * Sx;
        end

        ham_arr[1,:,2,:] = hs[idx] * Sz;
        
        if idx != 1
            ham_arr[2,:,2,:] = Id;
            ham_arr[3,:,2,:] = Sx;
        end
        
        mpos[idx] = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)

    end

    return DMRG_types.MPO([mpo for mpo in mpos])

end