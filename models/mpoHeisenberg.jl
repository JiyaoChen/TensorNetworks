function mpoHeisenberg(;J::Float64 = 1.0, spinS::Float64 = 1/2, setSym = "")

    # construct MPOs with different symmetries
    if setSym == ""

        # set vector spaces
        d = Int(2 * spinS + 1);
        vP = ℂ^d;
        vM = ℂ^5;

        # get spin operators
        Sx, Sy, Sz, Id = getSpinOperators(spinS);

        # generate the Heisenberg MPO
        ham_arr = zeros(Complex{Float64}, dim(vP), dim(vM), dim(vM), dim(vP));
        ham_arr[:,1,1,:] = Id;
        ham_arr[:,1,2,:] = J * Sx;
        ham_arr[:,1,3,:] = J * Sy;
        ham_arr[:,1,4,:] = J * Sz;
        ham_arr[:,2,5,:] = Sx;
        ham_arr[:,3,5,:] = Sy;
        ham_arr[:,4,5,:] = Sz;
        ham_arr[:,5,5,:] = Id;
        mpo = TensorMap(ham_arr, vP * vM, vM * vP)

    elseif setSym == "U1"

        # TODO

    elseif setSym == "SU2"

        # set vector spaces
        vP = SU₂Space(physicalSpin => 1);
        vV = SU₂Space(0 => 1, 1 / 2 => 1);
        vM = SU₂Space(0 => 2, 1 => 1);

        # construct empty MPO
        mpo = TensorMap(zeros, ComplexF64, vP ⊗ vM, vM ⊗ vP)

        # fill tensor blocks
        tensorDict = convert(Dict, fullOperator);
        dictKeys = keys(tensorDict);
        dataDict = tensorDict[:data];
        dataDict["Irrep[SU₂](1/2)"] = [1.0 + 0.0im 0.0 + 0.0im γ + 0.0im ; 0.0 + 0.0im 1.0 + 0.0im 0.0 + 0.0im ; 0.0 + 0.0im γ + 0.0im 0.0 + 0.0im];
        tensorDict[:data] = dataDict;
        mpo = convert(TensorMap, tensorDict);

    end

    # function return
    return mpo

end