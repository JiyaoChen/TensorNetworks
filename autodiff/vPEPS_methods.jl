function optimizePEPS(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)
   
    optimmethod = LBFGS(m = 20);
    optimargs = (Optim.Options(f_tol = 1e-6, show_trace = true), );
    res = nothing;
    let groundStateEnergy = x -> real(computeEnergy(x, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG))
        res = optimize(groundStateEnergy, Δ -> Zygote.gradient(groundStateEnergy, Δ)[1], pepsTensorsVec, optimmethod, inplace = false, optimargs...);
    end

    return res;

end

function computeEnergy(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod, energyTBG)

    # normalize pepsTensorsVec
    # shape = size(pepsTensorsVec)
    # pepsTensorsVec = reshape([ind < sqrt(truncBelowE) ? 0 : ind for ind in pepsTensorsVec], shape);
    # normalization = norm(pepsTensorsVec)
    # println("Norm of PEPS tensors: $normalization")
    # pepsTensorsVec /= norm(pepsTensorsVec);

    # reshape pepsTensor into array of iPEPS tensors
    Lx = size(pepsTensorsVec, 1);
    Ly = size(pepsTensorsVec, 2);
    pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] / norm(pepsTensorsVec[idx, idy, :, :, :, :, :]) for idx = 1 : Lx, idy = 1 : Ly];
    # normalization = norm(pepsTensorsVec)
    # println("Norm of PEPS tensors: $normalization")
    # pepsTensors = pepsTensorsVec;

    # run CTMRG
    CTMRGTensors = runCTMRG(pepsTensors, unitCellLayout, chiE, truncBelowE, convTolE, maxIter, initMethod);

    # compute energy
    gsE = energy(pepsTensors, unitCellLayout, CTMRGTensors, energyTBG);
    return gsE

end

# containing some auxiliary functions to work with the peps objects
# @Zygote.nograd getCoordinates
function getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout)
    
    # set ordering of tensors in the Lx × Ly unit cell
    tensorNumbers = reshape(collect(1 : Lx * Ly), Lx, Ly)
    
    # convert (latticeIdx, latticeIdy) to (unitCellIdx, unitCellIdy)
    unitCellLx, unitCellLy = size(unitCellLayout);
    unitCellIdx = mod(latticeIdx - 1, unitCellLx) + 1;
    unitCellIdy = mod(latticeIdy - 1, unitCellLy) + 1;

    # get number of tensor
    tensorNum = unitCellLayout[unitCellIdx, unitCellIdy];

    # get posX and posY of (latticeIdx, latticeIdy) in unit cell
    # tensorIdx = findfirst(tensorNumbers .== tensorNum);
    tensorIdx = CartesianIndices(tensorNumbers)[tensorNum];
    posX = tensorIdx[1];
    posY = tensorIdx[2];

    return (posX, posY);

end

# @Zygote.nograd toTensorNum
function toTensorNum(lx, Lx, ly, Ly)

    tensorNumbering = reshape(collect(1 : Lx * Ly), Lx, Ly);
    return tensorNumbering[lx, ly]

end

# Base.similar(UC::pepsUnitCell) = pepsUnitCell(UC.Lx, UC.Ly, Array{eltype(UC.tensorArray), 2}(undef, UC.Lx, UC.Ly), UC.unitCellLayout)

# function Base.getindex(UC::pepsUnitCell, latticeIdx::T, latticeIdy::T) where T <: Integer
#     Lx = UC.Lx;
#     Ly = UC.Ly;
#     tensorArray = UC.tensorArray;
#     unitCellLayout = UC.unitCellLayout;
#     posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
#     return tensorArray[posX, posY]
# end

# this would be a mutable function (which we want to avoid)
# function Base.setindex!(UC::pepsUnitCell, pepsTensor, latticeIdx::T, latticeIdy::T) where T <: Integer
#     Lx = UC.Lx;
#     Ly = UC.Ly;
#     tensorArray = UC.tensorArray;
#     unitCellLayout = UC.unitCellLayout;
#     posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
#     tensorArray[posX, posY] = pepsTensor;
#     return UC
# end

# initializers

# @Zygote.nograd initializeC
function initializeC(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        c = ones(elementType, 1, 1);
    elseif initMethod == 1
        c = ones(elementType, 1, 1);
    elseif initMethod == 2
        c = randn(elementType, 1, 1);
    elseif initMethod == 3
        c = randn(elementType, chiE, chiE);
    end
    return c
end

# @Zygote.nograd initializeT1
function initializeT1(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T1 = ones(elementType, 1, tensorDims[5], tensorDims[5], 1);
    elseif initMethod == 1
        T1 = reshape(Matrix{elementType}(I, tensorDims[5], tensorDims[5]), (1, tensorDims[5], tensorDims[5], 1));
    elseif initMethod == 2
        T1 = randn(elementType, 1, tensorDims[5], tensorDims[5], 1);
    elseif initMethod == 3
        T1 = randn(elementType, chiE, tensorDims[5], tensorDims[5], chiE);
    end
    return T1
end

# @Zygote.nograd initializeT2
function initializeT2(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T2 = ones(elementType, tensorDims[4], tensorDims[4], 1, 1);
    elseif initMethod == 1
        T2 = reshape(Matrix{elementType}(I, tensorDims[4], tensorDims[4]), (tensorDims[4], tensorDims[4], 1, 1));
    elseif initMethod == 2
        T2 = randn(elementType, tensorDims[4], tensorDims[4], 1, 1);
    elseif initMethod == 3
        T2 = randn(elementType, tensorDims[4], tensorDims[4], chiE, chiE);
    end
    return T2
end

# @Zygote.nograd initializeT3
function initializeT3(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T3 = ones(elementType, 1, 1, tensorDims[2], tensorDims[2]);
    elseif initMethod == 1
        T3 = reshape(Matrix{elementType}(I, tensorDims[2], tensorDims[2]), (1, 1, tensorDims[2], tensorDims[2]));
    elseif initMethod == 2
        T3 = randn(elementType, 1, 1, tensorDims[2], tensorDims[2]);
    elseif initMethod == 3
        T3 = randn(elementType, chiE, chiE, tensorDims[2], tensorDims[2]);
    end
    return T3
end

# @Zygote.nograd initializeT4
function initializeT4(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T4 = ones(elementType, 1, tensorDims[1], tensorDims[1], 1);
    elseif initMethod == 1
        T4 = reshape(Matrix{elementType}(I, tensorDims[1], tensorDims[1]), (1, tensorDims[1], tensorDims[1], 1));
    elseif initMethod == 2
        T4 = randn(elementType, 1, tensorDims[1], tensorDims[1], 1);
    elseif initMethod == 3
        T4 = randn(elementType, chiE, tensorDims[1], tensorDims[1], chiE);
    end
    return T4
end

# @Zygote.nograd initializeTensors
function initializeTensors(initFunc, elementType, dimensionsTensors, chiE::Int, initMethod::Int, T, Lx::Int, Ly::Int)
    
    # initialize array with CTM tensors
    # tensorArray = Array{T, 2}([initFunc(elementType, dimensionsTensors[idx, idy], chiE, initMethod) for idx = 1 : Lx, idy = 1 : Ly]);
    tensorArray = [initFunc(elementType, dimensionsTensors[idx, idy], chiE, initMethod) for idx = 1 : Lx, idy = 1 : Ly];
    return tensorArray

end

# @Zygote.nograd initializeCTMRGTensors
function initializeCTMRGTensors(elementType, dimensionsTensors, chiE::Int; initMethod::Int = 0)

    # get size
    Lx, Ly = size(dimensionsTensors);

    # set types for C and T tensors
    typeC = Array{elementType, 2};
    typeT = Array{elementType, 4};

    C1 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly);
    T1 = initializeTensors(initializeT1, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly);

    C2 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly);
    T2 = initializeTensors(initializeT2, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly);

    C3 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly);
    T3 = initializeTensors(initializeT3, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly);

    C4 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly);
    T4 = initializeTensors(initializeT4, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly);

    # return CTMRG tensors
    return C1, T1, C2, T2, C3, T3, C4, T4

end