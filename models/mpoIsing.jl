using TensorKit
include("getSpinOperators.jl")

function mpoIsing(; J::Float64 = 4.0, h::Float64 = 2.0, setSym::String = "Z2")

    # set vector spaces
    if setSym == ""
        vP = ℂ^2;
        vM = ℂ^3;
    elseif setSym == "Z2"
        vP = ℤ₂Space(0 => 1, 1 => 1);
        vV = ℤ₂Space(0 => 1);
        vM = ℤ₂Space(0 => 2, 1 => 1);
    end

    # get spin operators
    Sx, Sy, Sz, Id = getSpinOperators(1/2);

    # generate the Ising MPO
    ham_arr = zeros(ComplexF64, dim(vM), dim(vP), dim(vM), dim(vP));
    ham_arr[1,:,1,:] = Id;
    ham_arr[2,:,2,:] = Id;
    ham_arr[1,:,3,:] = J * Sx;
    ham_arr[3,:,2,:] = Sx;
    ham_arr[1,:,2,:] = h * Sz;
    mpo = TensorMap(ham_arr, vM ⊗ vP, vM ⊗ vP)

    # function return
    return mpo

end

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