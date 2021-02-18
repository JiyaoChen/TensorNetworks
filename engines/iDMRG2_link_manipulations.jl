function computeSharedLink(mpsSpaceL::E, physSpaceL::E, physSpaceR::E, mpsSpaceR::E) where E<:EuclideanSpace
    v1 = fuse(mpsSpaceL, physSpaceL)
    v2 = fuse(physSpaceR, mpsSpaceR)
    vM = infimum(v1, v2)
    return vM
end