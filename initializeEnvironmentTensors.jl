# initialize CTM tensors around the finitePEPS circumference
function initializeEnvironmentTensors(finitePEPS, finitePEPO)
    # initialize empty array
    envTensors = Array{Any,2}(undef, size(finitePEPS))

    # initialize vectorSpace along the perimeter
    vecSpaceTriv = ℂ^1;
    for idx = 1 : Lx
        for idy = 1 : Ly
            
            # get PEPS tensor
            bulkPEPS = finitePEPS[idx, idy];
            bulkPEPO = finitePEPO[idx, idy];

            # initialize CTM tensors
            C1 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T1 = TensorMap(ones, vecSpaceTriv ⊗ space(bulkPEPS,5)' ⊗ space(bulkPEPO,5)' ⊗ space(bulkPEPS,5), vecSpaceTriv);
            C2 = TensorMap(ones, vecSpaceTriv ⊗ vecSpaceTriv, one(vecSpaceTriv));
            T2 = TensorMap(ones, space(bulkPEPS,4)' ⊗ space(bulkPEPO,4)' ⊗ space(bulkPEPS,4) ⊗ vecSpaceTriv, vecSpaceTriv);
            C3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv ⊗ space(bulkPEPS,2)' ⊗ space(bulkPEPO,2) ⊗ space(bulkPEPS,2));
            C4 = TensorMap(ones, one(vecSpaceTriv), vecSpaceTriv ⊗ vecSpaceTriv);
            T4 = TensorMap(ones, vecSpaceTriv, space(bulkPEPS,1)' ⊗ space(bulkPEPO,1) ⊗ space(bulkPEPS,1) ⊗ vecSpaceTriv);
            
            # store CTM tensors
            envTensors[idx,idy] = [C1, T1, C2, T2, C3, T3, C4, T4];

        end
    end

    return envTensors
end