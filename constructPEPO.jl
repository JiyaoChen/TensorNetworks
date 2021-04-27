using TensorKit
include("models/getSpinOperators.jl")

function constructPEPOHeisenberg(P::Dict)
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])
    # set vector spaces
    vP = ℂ^Int64(floor(real(2*P["spin"]+1)))
    vM = ℂ^5

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    # fill in pepos
    pepos = Array{Any}(undef,P["Lx"],P["Ly"])
    Js = fill(1.0, size(pepos,1), size(pepos,2))
    for idx=1:size(pepos,1), idy=1:size(pepos,2)
        J = Js[idx,idy]
        ham_arr = zeros(P["eltype"], dim(vM), dim(vM), dim(vP), dim(vM), dim(vM), dim(vP))

        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx != size(pepos,1)
            ham_arr[1,2,:,1,2,:] = Id
            ham_arr[1,2,:,4,2,:] = 0.5 * J * Sm
            ham_arr[1,2,:,5,2,:] = 0.5 * J * Sp
            ham_arr[1,2,:,3,2,:] = J * Sz
        end
        if idy != size(pepos,2)
            ham_arr[2,1,:,2,1,:] = Id
            ham_arr[2,1,:,2,4,:] = 0.5 * J * Sm
            ham_arr[2,1,:,2,5,:] = 0.5 * J * Sp
            ham_arr[2,1,:,2,3,:] = J * Sz
        end
        if idx != 1
            ham_arr[4,2,:,2,2,:] = Sp
            ham_arr[5,2,:,2,2,:] = Sm
            ham_arr[3,2,:,2,2,:] = Sz
            ham_arr[2,2,:,2,2,:] = Id
        end
        if idy != 1
            ham_arr[2,4,:,2,2,:] = Sp
            ham_arr[2,5,:,2,2,:] = Sm
            ham_arr[2,3,:,2,2,:] = Sz
            ham_arr[2,2,:,2,2,:] = Id
        end
        pepos[idx, idy] = TensorMap(ham_arr, vM ⊗ vM ⊗ vP, vM ⊗ vM ⊗ vP)

    end
    return pepos
end

function constructPEPOIdentity(P::Dict)
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])
    # set vector spaces
    vP = ℂ^Int64(floor(real(2*P["spin"]+1)))
    vM = ℂ^1

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    # fill in pepos
    pepos = Array{Any}(undef,P["Lx"],P["Ly"])
    for idx=1:size(pepos,1), idy=1:size(pepos,2)
        ham_arr = zeros(P["eltype"], dim(vM), dim(vM), dim(vP), dim(vM), dim(vM), dim(vP))
        ham_arr[1,1,:,1,1,:] = Id
        pepos[idx, idy] = TensorMap(ham_arr, vM ⊗ vM ⊗ vP, vM ⊗ vM ⊗ vP)
    end
    return pepos
end

function constructPEPOIsing(P::Dict)
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])
    # set vector spaces
    vP = ℂ^Int64(floor(real(2*P["spin"]+1)))
    vM = ℂ^3

    # get spin operators
    Sx, Sy, Sz, Id, Sm, Sp = getSpinOperators(P["spin"])

    # fill in pepos
    pepos = Array{Any}(undef,P["Lx"],P["Ly"])
    Js = fill(1.0, size(pepos,1), size(pepos,2))
    Bs = fill(1.0, size(pepos,1), size(pepos,2))
    for idx=1:size(pepos,1), idy=1:size(pepos,2)
        J = Js[idx,idy]
        B = Bs[idx,idy]
        ham_arr = zeros(P["eltype"], dim(vM), dim(vM), dim(vP), dim(vM), dim(vM), dim(vP))

        # let's spare ourselves the trouble of defining boundary vectors :)
        if idx != size(pepos,1)
            ham_arr[1,2,:,1,2,:] = Id
            ham_arr[1,2,:,3,2,:] = J * Sx
        end
        if idy != size(pepos,2)
            ham_arr[2,1,:,2,1,:] = Id
            ham_arr[2,1,:,2,3,:] = J * Sx
        end
        if idx != 1
            ham_arr[1,2,:,2,2,:] = B * Sz
            ham_arr[3,2,:,2,2,:] = Sx
            ham_arr[2,2,:,2,2,:] = Id
        end
        if idy != 1
            ham_arr[2,1,:,2,2,:] = B * Sz
            ham_arr[2,3,:,2,2,:] = Sx
            ham_arr[2,2,:,2,2,:] = Id
        end
        pepos[idx, idy] = TensorMap(ham_arr, vM ⊗ vM ⊗ vP, vM ⊗ vM ⊗ vP)

    end
    return pepos
end