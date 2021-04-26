# initialize CTM tensors around the finitePEPS circumference
function initializeEnvironmentTensors(finitePEPS, finitePEPO)
    envTensors = Array{Array{Any,1},2}(undef, Lx, Ly);

    # initialize vectorSpace along the perimeter
    vecSpaceTriv = ℂ^1;
    for idx = 1 : Lx
        for idy = 1 : Ly
            
            # get PEPS tensor
            bulkTensor = finitePEPS[idx, idy];
            bulkPEPO = finitePEPO[idx, idy];

            # initialize CTM tensors
            C1 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T1 = TensorMap(ones, vecSpaceTriv ⊗ space(bulkTensor,5)' ⊗ space(bulkPEPO,5)' ⊗ space(bulkTensor,5), vecSpaceTriv);
            C2 = TensorMap(ones, vecSpaceTriv ⊗ vecSpaceTriv, one(vecSpaceTriv));
            T2 = TensorMap(ones, space(bulkTensor,4)' ⊗ space(bulkPEPO,4)' ⊗ space(bulkTensor,4) ⊗ vecSpaceTriv, vecSpaceTriv);
            C3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv ⊗ space(bulkTensor,2)' ⊗ space(bulkPEPO,2)' ⊗ space(bulkTensor,2));
            C4 = TensorMap(ones, one(vecSpaceTriv), vecSpaceTriv ⊗ vecSpaceTriv);
            T4 = TensorMap(ones, vecSpaceTriv, space(bulkTensor,1)' ⊗ space(bulkTensor,1) ⊗ vecSpaceTriv);
            
            # store CTM tensors
            envTensors[idx,idy] = [C1, T1, C2, T2, C3, T3, C4, T4];

        end
    end
end