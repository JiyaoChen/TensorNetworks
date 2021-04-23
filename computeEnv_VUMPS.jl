
using JLD
using LinearAlgebra
using Printf
using SpecialFunctions
using TensorKit
using WignerSymbols

# clear console
Base.run(`clear`)

# set chiB
chiB = 2;

# set betaT
betaT = 1.0;

# rescale Bessel factors
doRescaling = 0;

# set quantum numbers on virtual indices
maxSpin = chiB - 1;
quantumNumbersVirt = collect(0 : maxSpin);
vecSpaceVirt = SU2Space([j => 1 for j in quantumNumbersVirt]);

# conversion factor between spherical harmonics and Clebsch-Gordan coefficients
function factorSHCG(l1,l2,l3)
    conversionFactor = sqrt((2 * l1 + 1) * (2 * l2 + 1) / (4 * pi * (2 * l3 + 1))) * clebschgordan(l1,0,l2,0,l3,0);
    return conversionFactor;
end

# function to generate chargeSectors for regular partition function tensor
function generateChargeSectorsInOut_PF(quantNum1, quantNum2, quantNum3, quantNum4)

    # determine different fusionSectorsOutgoing
    fusionSectorsOutgoing = Array{Int64,2}(undef,0,3);
    for ida = quantNum2, idb = quantNum1
        for idf = collect(abs(ida - idb) : 1 : abs(ida + idb))
            fusionSectorsOutgoing = vcat(fusionSectorsOutgoing, [idb ida idf]);
        end
    end

    # determine different fusionSectorsIncoming
    fusionSectorsIncoming = Array{Int64,2}(undef,0,3);
    for ida = quantNum4, idb = quantNum3
        for idf = collect(abs(ida - idb) : 1 : abs(ida + idb))
            fusionSectorsIncoming = vcat(fusionSectorsIncoming, [idb ida idf]);
        end
    end

    # find unique quantum numbers on fused index
    uniqueSectors = intersect(fusionSectorsOutgoing[:,3],fusionSectorsIncoming[:,3])
    return fusionSectorsOutgoing, fusionSectorsIncoming, uniqueSectors;

end


#---------------------------------------------------------------------------------------------------------------------------------------
# construction for regular partition function tensor
#---------------------------------------------------------------------------------------------------------------------------------------

# get chargeSectors for regular partition function tensors
chargeSectorsOutgoing_PF, chargeSectorsIncoming_PF, uniqueSectors_PF = generateChargeSectorsInOut_PF(quantumNumbersVirt, quantumNumbersVirt, quantumNumbersVirt, quantumNumbersVirt);

# initialize tensor for the partition function of the Heisenberg model
tensorPartitionFunction = TensorMap(zeros, ComplexF64, vecSpaceVirt ⊗ vecSpaceVirt, vecSpaceVirt ⊗ vecSpaceVirt);
tensorDictPartitionFunction = convert(Dict, tensorPartitionFunction);
tensorBlocks = tensorDictPartitionFunction[:data]

# compute and assign degeneracyTensors to tensorBlocks
for secIdx_PF = 1 : length(uniqueSectors_PF)

    # get internal quantum number
    qNumInt_PF = uniqueSectors_PF[secIdx_PF];

    # find external quantum numbers that yield the interal quantum number
    sectorsOutgoing_PF = chargeSectorsOutgoing_PF[chargeSectorsOutgoing_PF[:,3] .== qNumInt_PF,:];
    sectorsIncoming_PF = chargeSectorsIncoming_PF[chargeSectorsIncoming_PF[:,3] .== qNumInt_PF,:];
    
    # initialize degeneracyTensor_PF
    dimDegeneracyTensor_PF = [ size(sectorsOutgoing_PF,1) , size(sectorsIncoming_PF,1) ];
    degeneracyTensor_PF = zeros(ComplexF64, dimDegeneracyTensor_PF[1], dimDegeneracyTensor_PF[2]);

    # fill degeneracyTensor_PF
    for idx = 1 : dimDegeneracyTensor_PF[1], idy = 1 : dimDegeneracyTensor_PF[2]

        irrepsOutgoing = sectorsOutgoing_PF[idx,:];
        irrepsIncoming = sectorsIncoming_PF[idy,:];
        sectorQuantumNumbers = [irrepsOutgoing[1 : 2] irrepsIncoming[1 : 2]];
        
        # construct numerical factors
        besselFactors_PF = Vector{ComplexF64}(undef,4);
        for besselIdx = 1 : length(besselFactors_PF)
            besselFactors_PF[besselIdx] = sqrt( sqrt(pi / (2*betaT)) * besseli(sectorQuantumNumbers[besselIdx] + 1/2,betaT) );
        end
        numFactor = 1im^sum(sectorQuantumNumbers) * prod(besselFactors_PF);

        # apply rescaling of Bessel factors
        if doRescaling == 1
            numFactor = numFactor / besseli(1/2,betaT);
        end

        # account for conversion factor between spherical harmonics and Clebsch-Gordan coefficients
        numFactor *= (factorSHCG(irrepsOutgoing[1], irrepsOutgoing[2], qNumInt_PF) * factorSHCG(irrepsIncoming[1], irrepsIncoming[2], qNumInt_PF));

        # fill degeneracyTensor_PF
        degeneracyTensor_PF[idx,idy] = numFactor;

    end

    # set degeneracyTensor_PF
    irrepStr = "Irrep[SU₂](" * string(qNumInt_PF) * ")";
    tensorBlocks[irrepStr] = Array{ComplexF64}(degeneracyTensor_PF);

end
tensorDictPartitionFunction[:data] = tensorBlocks;
tensorPartitionFunction = convert(TensorMap, tensorDictPartitionFunction);


#---------------------------------------------------------------------------------------------------------------------------------------
# find fixed-point for linear transfer matrix T
#---------------------------------------------------------------------------------------------------------------------------------------

include("mps.jl")

chiE = 50;
vecSpaceMPS_N = SU2Space(0 => 1);
MPS = TensorMap(randn, ComplexF64, vecSpaceMPS_N ⊗ vecSpaceVirt, vecSpaceMPS_N);
MPO = tensorPartitionFunction;
λ, AL, C, AR, FL, FR = vumps(MPS, MPO, chiE; tol = 1e-10);

println(space(C))

