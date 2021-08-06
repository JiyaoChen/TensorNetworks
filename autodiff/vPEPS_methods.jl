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
function initializeC(tensorType, χ, initMethod::Int)
    if initMethod == 0
        c = ones(tensorType, 1, 1);
    elseif initMethod == 1
        c = randn(tensorType, χ, χ);
    end
    return c
end

# @Zygote.nograd initializeT1
function initializeT1(tensorType, chiB, χ, initMethod::Int)
    if initMethod == 0
        T1 = ones(tensorType, 1, chiB, chiB, 1);
    elseif initMethod == 1
        T1 = randn(tensorType, χ, chiB, chiB, χ);
    end
    return T1
end

# @Zygote.nograd initializeT2
function initializeT2(tensorType, chiB, χ, initMethod::Int)
    if initMethod == 0
        T2 = ones(tensorType, chiB, chiB, 1, 1);
    elseif initMethod == 1
        T2 = randn(tensorType, chiB, chiB, χ, χ);
    end
    return T2
end

# @Zygote.nograd initializeT3
function initializeT3(tensorType, chiB, χ, initMethod::Int)
    if initMethod == 0
        T3 = ones(tensorType, 1, 1, chiB, chiB);
    elseif initMethod == 1
        T3 = randn(tensorType, χ, χ, chiB, chiB);
    end
    return T3
end

# @Zygote.nograd initializeT4
function initializeT4(tensorType, chiB, χ, initMethod::Int)
    if initMethod == 0
        T4 = ones(tensorType, 1, chiB, chiB, 1);
    elseif initMethod == 1
        T4 = randn(tensorType, χ, chiB, chiB, χ);
    end
    return T4
end

# @Zygote.nograd initializeTensors
function initializeTensors(initFunc, tensorArray::Array, χ, initMethod::Int, T, Lx, Ly, unitCellLayout)
    
    # initialize array with CTM tensors
    tensorArray = Array{T, 2}([initFunc(tensorArray[idx, idy], χ, initMethod) for idx = 1 : Lx, idy = 1 : Ly]);
    return tensorArray

end

# @Zygote.nograd initializeCTMRGTensors
function initializeCTMRGTensors(pepsTensors, unitCellLayout, chiE::Int; initMethod = 0)
    
    # get size
    Lx, Ly = size(pepsTensors);

    # set types for C and T tensors
    tensorType = eltype(eltype(pepsTensors));
    typeC = Array{tensorType, 2};
    typeT = Array{tensorType, 4};

    C1 = initializeTensors(initializeC, tensorType, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T1 = initializeTensors(initializeT1, pepsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C2 = initializeTensors(initializeC, tensorType, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T2 = initializeTensors(initializeT2, pepsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C3 = initializeTensors(initializeC, tensorType, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T3 = initializeTensors(initializeT3, pepsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C4 = initializeTensors(initializeC, tensorType, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T4 = initializeTensors(initializeT4, pepsTensors, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    # return CTMRG tensors
    return C1, T1, C2, T2, C3, T3, C4, T4

end