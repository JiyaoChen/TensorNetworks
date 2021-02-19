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