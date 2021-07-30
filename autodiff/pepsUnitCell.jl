# the basic struct used to define a tensorial peps object on a given unit cell
mutable struct pepsUnitCell{T}
    Lx::Int64
    Ly::Int64
    tensorDict::Dict{Tuple{Int, Int}, T}
    unitCellLayout::Matrix{Int64}
end

# containing some auxiliary functions to work with the peps objects
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
    tensorIdx = findfirst(tensorNumbers .== tensorNum);
    posX = tensorIdx[1];
    posY = tensorIdx[2];
    return posX, posY;

end

Base.similar(UC::pepsUnitCell) = pepsUnitCell(UC.Lx, UC.Ly, typeof(UC.tensorDict)(), UC.unitCellLayout)

function Base.getindex(UC::pepsUnitCell, latticeIdx::T, latticeIdy::T) where T <: Integer
    Lx = UC.Lx;
    Ly = UC.Ly;
    tensorDict = UC.tensorDict;
    unitCellLayout = UC.unitCellLayout;
    posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
    return tensorDict[(posX, posY)]
end

function Base.setindex!(UC::pepsUnitCell, pepsTensor, latticeIdx::T, latticeIdy::T) where T <: Integer
    Lx = UC.Lx;
    Ly = UC.Ly;
    tensorDict = UC.tensorDict;
    unitCellLayout = UC.unitCellLayout;
    posX, posY = getCoordinates(latticeIdx, Lx, latticeIdy, Ly, unitCellLayout);
    tensorDict[(posX, posY)] = pepsTensor;
    return UC
end

# initializers
function initializeC(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        c = ones(eltype(pepsTensor), 1, 1);
    elseif initMethod == 1
        c = randn(eltype(pepsTensor), χ, χ);
    end
    return c
end

function initializeT1(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T1 = ones(eltype(pepsTensor), 1, size(pepsTensor, 5), size(pepsTensor, 5), 1);
    elseif initMethod == 1
        T1 = randn(eltype(pepsTensor), χ, size(pepsTensor, 5), size(pepsTensor, 5), χ);
    end
    return T1
end

function initializeT2(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T2 = ones(eltype(pepsTensor), size(pepsTensor, 4), size(pepsTensor, 4), 1, 1);
    elseif initMethod == 1
        T2 = randn(eltype(pepsTensor), size(pepsTensor, 4), size(pepsTensor, 4), χ, χ);
    end
    return T2
end

function initializeT3(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T3 = ones(eltype(pepsTensor), 1, 1, size(pepsTensor, 2), size(pepsTensor, 2));
    elseif initMethod == 1
        T3 = randn(eltype(pepsTensor), χ, χ, size(pepsTensor, 2), size(pepsTensor, 2));
    end
    return T3
end

function initializeT4(pepsTensor, χ, initMethod::Int)
    if initMethod == 0
        T4 = ones(eltype(pepsTensor), 1, size(pepsTensor, 1), size(pepsTensor, 1), 1);
    elseif initMethod == 1
        T4 = randn(eltype(pepsTensor), χ, size(pepsTensor, 1), size(pepsTensor, 1), χ);
    end
    return T4
end


function initializeTensors(initFunc, tensorDict::Dict, χ, initMethod::Int, T, Lx, Ly, unitCellLayout)
    
    # initialize empty struct for unitCell
    unitCell = pepsUnitCell(Lx, Ly, Dict{Tuple{Int, Int}, T}(), unitCellLayout);

    # call initFunc for each element in the tensorDict Dict and return unitCell
    foreach(u -> unitCell[u...] = initFunc(tensorDict[u], χ, initMethod), keys(tensorDict));
    return unitCell

end

function initializeCTMRGTensors(iPEPS::pepsUnitCell, chiE::Int; initMethod = 0)
    
    # get struct variables
    Lx = iPEPS.Lx;
    Ly = iPEPS.Ly;
    tensorDict = iPEPS.tensorDict;
    unitCellLayout = iPEPS.unitCellLayout;

    # set types for C and T tensors
    typeC = Array{eltype(eltype(values(tensorDict))), 2};
    typeT = Array{eltype(eltype(values(tensorDict))), 4};

    C1 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T1 = initializeTensors(initializeT1, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C2 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T2 = initializeTensors(initializeT2, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C3 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T3 = initializeTensors(initializeT3, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    C4 = initializeTensors(initializeC, tensorDict, chiE, initMethod, typeC, Lx, Ly, unitCellLayout);
    T4 = initializeTensors(initializeT4, tensorDict, chiE, initMethod, typeT, Lx, Ly, unitCellLayout);

    # return CTMRG tensors
    return C1, T1, C2, T2, C3, T3, C4, T4

end