function computeSharedLink(mpsSpaceL::E, physSpaceL::E, physSpaceR::E, mpsSpaceR::E) where E<:EuclideanSpace
    v1 = fuse(mpsSpaceL, physSpaceL)
    v2 = fuse(physSpaceR, mpsSpaceR)
    vM = infimum(v1, v2)
    return vM
end

function shiftVirtSpaceMPS(mps::A, vecSpace::S) where A<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace
    mpsSpaces = space(mps)
    fusedSpace = fuse(vecSpace, mpsSpaces[3]')
    idTensor = TensorKit.isomorphism(vecSpace ⊗ mpsSpaces[3]', fusedSpace)

    newTensor = permute(mps * permute(idTensor, (2,), (1, 3)), (3,1), (2,4))
    newTensor = braid(newTensor, )
    braid(t::AbstractTensorMap{S,N₁,N₂}, levels::NTuple{2+2,Int},
          p1::NTuple{1,Int}, p2::NTuple{2,Int})
    permute(t::AbstractTensorMap{S,N₁,N₂},
            p1::NTuple{N₁′,Int}, p2::NTuple{N₂′,Int}; copy = false)
    println(newTensor)
    # println(mps)
    # ), (1*3), (2,4))
    return mps
end

function shiftIntegerIrrepsEnv(env::A, sector::Int64) where A<:AbstractTensorMap{S,1,2} where S<:EuclideanSpace

    return env
end