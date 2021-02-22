function computeSharedLink(mpsSpaceL::E, physSpaceL::E, physSpaceR::E, mpsSpaceR::E) where E<:EuclideanSpace
    v1 = fuse(mpsSpaceL, physSpaceL)
    v2 = fuse(physSpaceR, mpsSpaceR)
    vM = infimum(v1, v2)
    return vM
end

function shiftVirtSpaceMPS(mps::A, vecSpace::S) where A<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace
    
    # get vector spaces of MPS tensor
    mpsSpaces = space(mps)

    # construct isomorphisms for left and right indices
    fusedSpaceL = fuse(mpsSpaces[1], vecSpace)
    fusedSpaceR = fuse(vecSpace, mpsSpaces[3]')
    idTensorL = TensorKit.isomorphism(fusedSpaceL, mpsSpaces[1] ⊗ vecSpace)
    idTensorR = TensorKit.isomorphism(vecSpace ⊗ mpsSpaces[3]', fusedSpaceR)

    @tensor mps[-1 -2; -3] := idTensorL[-1 1 2] * mps[1 -2 3] * idTensorR[2 3 -3]
    return mps
end
