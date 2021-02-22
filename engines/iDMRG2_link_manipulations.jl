function computeSharedLink(mpsSpaceL::E, physSpaceL::E, physSpaceR::E, mpsSpaceR::E) where E<:EuclideanSpace
    v1 = fuse(mpsSpaceL, physSpaceL)
    v2 = fuse(physSpaceR, mpsSpaceR)
    vM = infimum(v1, v2)
    return vM
end

function shiftVirtSpaceMPS(mps::A, shiftIrrep::S, maxIrrepL::S, maxIrrepR::S) where A<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace
    
    # get vector spaces of MPS tensor
    mpsSpaces = space(mps)

    # construct isomorphisms for left and right indices
    fusedSpaceL = fuse(mpsSpaces[1], shiftIrrep)
    fusedSpaceR = fuse(shiftIrrep, mpsSpaces[3]')
    idTensorL = TensorKit.isomorphism(fusedSpaceL, mpsSpaces[1] ⊗ shiftIrrep)
    idTensorR = TensorKit.isomorphism(shiftIrrep ⊗ mpsSpaces[3]', fusedSpaceR)

    # construct truncation isometries for left and right indices
    truncIsometryL = TensorKit.isometry(fusedSpaceL, maxIrrepL)'
    truncIsometryR = TensorKit.isometry(fusedSpaceR, maxIrrepR)

    # construct new environment tensor
    @tensor mps[-1 -2; -3] := idTensorL[-1 1 2] * mps[1 -2 3] * idTensorR[2 3 4] * truncIsometryR[4 -3]
    # @tensor mps[-1 -2; -3] := truncIsometryL[-1 1] * idTensorL[1 2 3] * mps[2 -2 4] * idTensorR[3 4 5] * truncIsometryR[5 -3]
    return mps
end

function shiftVirtSpaceMPS(mps::A, shiftIrrep::S) where A<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace
    
    # get vector spaces of MPS tensor
    mpsSpaces = space(mps)

    # construct isomorphisms for left and right indices
    fusedSpaceL = fuse(mpsSpaces[1], shiftIrrep)
    fusedSpaceR = fuse(shiftIrrep, mpsSpaces[3]')
    idTensorL = TensorKit.isomorphism(fusedSpaceL, mpsSpaces[1] ⊗ shiftIrrep)
    idTensorR = TensorKit.isomorphism(shiftIrrep ⊗ mpsSpaces[3]', fusedSpaceR)

    # construct new environment tensor
    @tensor mps[-1 -2; -3] := idTensorL[-1 1 2] * mps[1 -2 3] * idTensorR[2 3 -3]
    # @tensor mps[-1 -2; -3] := truncIsometryL[-1 1] * idTensorL[1 2 3] * mps[2 -2 4] * idTensorR[3 4 5] * truncIsometryR[5 -3]
    return mps
end

function shiftVirtSpaceEnv(env::A, shiftIrrep::S, maxIrrep::S) where A<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace
    
    # get vector spaces of environment tensor
    envSpaces = space(env)
    # print(envSpaces,"\n")

    # construct isomorphisms for left and right indices
    fusedSpaceL = fuse(envSpaces[1], shiftIrrep)
    fusedSpaceR = fuse(shiftIrrep, envSpaces[3]')
    idTensorL = TensorKit.isomorphism(fusedSpaceL, envSpaces[1] ⊗ shiftIrrep)
    idTensorR = TensorKit.isomorphism(shiftIrrep ⊗ envSpaces[3]', fusedSpaceR)

    # construct truncation isometries for left and right indices
    truncIsometryL = TensorKit.isometry(fusedSpaceL, maxIrrep)'
    # print(truncIsometryL)
    truncIsometryR = TensorKit.isometry(fusedSpaceR, maxIrrep)

    # construct new environment tensor
    # @tensor env[-1 -2; -3] := truncIsometryL[-1 1] * idTensorL[1 2 3] * env[2 -2 4] * idTensorR[3 4 5] * truncIsometryR[5 -3]
    @tensor env[-1 -2; -3] := idTensorL[-1 1 2] * env[1 -2 3] * idTensorR[2 3 4] * truncIsometryR[4 -3]
    # @tensor env[-1 -2; -3] := idTensorL[-1 1 2] * env[1 -2 3] * idTensorR[2 3 -3]
    println(space(env))
    println(space(truncIsometryL))
    println(space(truncIsometryR))
    return env
end