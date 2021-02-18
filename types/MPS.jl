# the struct is treated as a const which cannot be replaced during runtime --> if attributes change, restart REPL
mutable struct MPS{A<:AbstractTensorMap{S,2,1} where {S<:EuclideanSpace}}
    ALs::Vector{Union{Missing,A}}
    ACs::Vector{Union{Missing,A}}
    ARs::Vector{Union{Missing,A}}
end

# constructor for initialization with tensor train
function MPS(site_tensors::Vector{A}) where {A<:AbstractTensorMap{S,2,1} where {S<:EuclideanSpace}}

    for i in length(site_tensors) : -1 : 2
        (L, Q) = rightorth(site_tensors[i],(2,),(1,3),alg=LQpos())
        site_tensors[i] = permute(Q,(2,1),(3,))
        site_tensors[i-1] = site_tensors[i-1]*L
    end
    
    # initialize empty vectors
    ALs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ACs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ARs = Vector{Union{Missing,A}}(missing,length(site_tensors))

    ACs[1] = site_tensors[1]
    ARs[2:end] .= site_tensors[2:end]

    # call the default constructor and assign its attributes
    MPS{A}(ALs, ACs, ARs)
end