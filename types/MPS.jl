# the struct is treated as a const which cannot be replaced during runtime --> if attributes change, restart REPL
mutable struct MPS{A<:MPSTensor}
    ALs::Vector{A}
    ACs::Vector{A}
    ARs::Vector{A}
end

# constructor for initialization with tensor train
function MPS(site_tensors::Vector{A}) where {A<:MPSTensor}

    # bring MPS into right-canonical form
    for i in length(site_tensors) : -1 : 2
        (L, Q) = rightorth(site_tensors[i],(1,),(2,3),alg=LQpos())
        site_tensors[i] = permute(Q, (1,2), (3,))
        site_tensors[i-1] = site_tensors[i-1] * L
    end
    # (L, Q) = rightorth(site_tensors[1],(1,),(2,3),alg=LQpos())
    # site_tensors[1] = permute(Q, (1,2), (3,))

    # # check norm
    # norm = @tensor site_tensors[1][1,2,3]*conj(site_tensors[1][1,2,3])
    # println(norm)
    
    # initialize empty vectors
    ALs = Vector{A}(undef,length(site_tensors))
    ACs = Vector{A}(undef,length(site_tensors))
    ARs = Vector{A}(undef,length(site_tensors))

    norm = @tensor site_tensors[1][1 2 3] * conj(site_tensors[1][1 2 3])
    site_tensors[1] = site_tensors[1]/sqrt(norm)

    @tensor norm[:] := site_tensors[1][1 2 -1] * conj(site_tensors[1][1 2 -2])
    for i = 2 : length(ARs) - 1
        @tensor norm[:] := norm[1 2] * site_tensors[i][1 3 -1] * conj(site_tensors[i][2 3 -2])
    end
    norm = @tensor norm[1 2] * site_tensors[end][1 3 4] * conj(site_tensors[end][2 3 4])
    if abs(norm - 1e0) > 1e-12
        println(norm)
        throw(ErrorException("Norm of MPS not sufficiently close to 1.0. Norm is: " * string(norm)))
    end

    # @tensor norm[-1; -2] := site_tensors[1][1 2 -2] * conj(site_tensors[1][1 2 -1])
    # for i = 2 : length(ARs)
    #     @tensor norm[-1; -2] = norm[1 3] * site_tensors[i][3 2 -2] * conj(site_tensors[i][1 2 -1])
    # end
    # norm = trace(norm)
    # println(norm)

    ACs[1] = site_tensors[1]
    ARs[2:end] .= site_tensors[2:end]

    # call the default constructor and assign its attributes
    MPS{A}(ALs, ACs, ARs)

end

function MPS(model::Model ; init::Function = randn)
    
    # extract list of physical spaces from the MPO
    physSpaces = [space(mpo,4)' for mpo in model.H.mpo]
    # the virtual spaces are stored in the Q attribute of model
    virtSpaces = model.Q
    # generate the MPS tensor train with elements generated by init()
    mps = [TensorMap(init, model.P["eltype"], virtSpaces[i] ⊗ physSpaces[i], virtSpaces[i+1]) for i = 1 : length(physSpaces)]
    MPS(mps)

end