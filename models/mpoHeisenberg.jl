using TensorKit

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
    
    # elseif setSym == "SU2"

    #     # TODO this does not work with the VSCode debugger...

    #     # set vector spaces 
    #     vP = SU₂Space(spinS => 1)
    #     mpoSpaceL = SU₂Space(0 => 2, 1 => 1)
    #     mpoSpaceR = SU₂Space(0 => 2, 1 => 1)
    #     γ = sqrt(spinS * (spinS + 1))

    #     # construct empty MPO
    #     mpo = TensorMap(zeros, ComplexF64, mpoSpaceL ⊗ vP, mpoSpaceR ⊗ vP)

    #     # fill tensor blocks
    #     tensorDict = convert(Dict, mpo)
    #     dictKeys = keys(tensorDict)
    #     dataDict = tensorDict[:data]
    #     dataDict["Irrep[SU₂](1/2)"] = Array{ComplexF64}([1.0 0.0 J * γ ; 0.0 1.0 0.0 ; 0.0 γ 0.0])
    #     tensorDict[:data] = dataDict
    #     mpo = convert(TensorMap, tensorDict)

    end

    # function return
    return mpo

end