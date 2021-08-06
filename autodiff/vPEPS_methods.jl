function optimizePEPS(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)
   
    optimmethod = LBFGS(m = 20);
    optimargs = (Optim.Options(f_tol = 1e-6, show_trace = true), );
    res = nothing
    let energy = x -> real(computeEnergy(x, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG))
        res = optimize(energy, Δ -> Zygote.gradient(energy, Δ)[1], pepsTensors, optimmethod, inplace = false, optimargs...);
    end

    return res;

end

function computeEnergy(pepsTensorsVec, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod, energyTBG)

    # reshape pepsTensor into array of iPEPS tensors
    Lx = size(pepsTensorsVec, 1);
    Ly = size(pepsTensorsVec, 2);
    pepsTensors = [pepsTensorsVec[idx, idy, :, :, :, :, :] for idx = 1 : Lx, idy = 1 : Ly];

    # run CTMRG
    CTMRGTensors = runCTMRG(pepsTensors, unitCellLayout, chiE, truncBelowE, convTol, maxIter, initMethod);

    # compute energy
    return energy(pepsTensors, unitCellLayout, CTMRGTensors, energyTBG)

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
    return posX, posY;

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
        c = randn(elementType, chiE, chiE);
    end
    return c
end

# @Zygote.nograd initializeT1
function initializeT1(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T1 = ones(elementType, 1, tensorDims[5], tensorDims[5], 1);
    elseif initMethod == 1
        T1 = randn(elementType, chiE, tensorDims[5], tensorDims[5], chiE);
    end
    return T1
end

# @Zygote.nograd initializeT2
function initializeT2(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T2 = ones(elementType, tensorDims[4], tensorDims[4], 1, 1);
    elseif initMethod == 1
        T2 = randn(elementType, tensorDims[4], tensorDims[4], chiE, chiE);
    end
    return T2
end

# @Zygote.nograd initializeT3
function initializeT3(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T3 = ones(elementType, 1, 1, tensorDims[2], tensorDims[2]);
    elseif initMethod == 1
        T3 = randn(elementType, chiE, chiE, tensorDims[2], tensorDims[2]);
    end
    return T3
end

# @Zygote.nograd initializeT4
function initializeT4(elementType::DataType, tensorDims::NTuple{5, Int64}, chiE::Int, initMethod::Int)
    if initMethod == 0
        T4 = ones(elementType, 1, tensorDims[1], tensorDims[1], 1);
    elseif initMethod == 1
        T4 = randn(elementType, chiE, tensorDims[1], tensorDims[1], chiE);
    end
    return T4
end

# @Zygote.nograd initializeTensors
function initializeTensors(initFunc, elementType, dimensionsTensors, chiE, initMethod::Int, T, Lx, Ly, unitCellLayout)
    
    # initialize array with CTM tensors
    tensorArray = Array{T, 2}([initFunc(elementType, dimensionsTensors[idx, idy], chiE, initMethod) for idx = 1 : Lx, idy = 1 : Ly]);
    return tensorArray

end

# @Zygote.nograd initializeCTMRGTensors
function initializeCTMRGTensors(elementType, dimensionsTensors, unitCellLayout, chiE::Int; initMethod = 0)
    
    # get size
    Lx, Ly = size(dimensionsTensors);

    # set types for C and T tensors
    typeC = Array{elementType, 2};
    typeT = Array{elementType, 4};

    C1 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T1 = initializeTensors(initializeT1, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C2 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T2 = initializeTensors(initializeT2, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C3 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T3 = initializeTensors(initializeT3, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C4 = initializeTensors(initializeC, elementType, dimensionsTensors, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T4 = initializeTensors(initializeT4, elementType, dimensionsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    # return CTMRG tensors
    return C1, T1, C2, T2, C3, T3, C4, T4

end