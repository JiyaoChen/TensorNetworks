function initializeFinitePEPS(Lx, Ly, vecSpacePhys, vecSpaceVirt, vecSpaceTriv)

    # initialize finitePEPS
    finitePEPS = Array{PEPSType{S} where S<:EuclideanSpace,2}(undef, Lx, Ly);
    for idx = 1 : Lx, idy = 1 : Ly
        if idx == 1
            if idy == 1
                codomain = vecSpaceTriv ⊗ vecSpaceVirt;
                domain = vecSpaceVirt ⊗ vecSpaceTriv;
            elseif idy == Ly
                codomain = vecSpaceVirt ⊗ vecSpaceVirt;
                domain = vecSpaceTriv ⊗ vecSpaceTriv;
            else
                codomain = vecSpaceVirt ⊗ vecSpaceVirt;
                domain = vecSpaceVirt ⊗ vecSpaceTriv;
            end
        elseif idx == Lx
            if idy == 1
                codomain = vecSpaceTriv ⊗ vecSpaceTriv;
                domain = vecSpaceVirt ⊗ vecSpaceVirt;
            elseif idy == Ly
                codomain = vecSpaceVirt ⊗ vecSpaceTriv;
                domain = vecSpaceTriv ⊗ vecSpaceVirt;
            else
                codomain = vecSpaceVirt ⊗ vecSpaceTriv;
                domain = vecSpaceVirt ⊗ vecSpaceVirt;
            end
        else
            if idy == 1
                codomain = vecSpaceTriv ⊗ vecSpaceVirt;
                domain = vecSpaceVirt ⊗ vecSpaceVirt;
            elseif idy == Ly
                codomain = vecSpaceVirt ⊗ vecSpaceVirt;
                domain = vecSpaceTriv ⊗ vecSpaceVirt
            else
                codomain = vecSpaceVirt ⊗ vecSpaceVirt;
                domain = vecSpaceVirt ⊗ vecSpaceVirt;
            end
        end

        # append physical index
        codomain = codomain ⊗ vecSpacePhys;
        finitePEPS[idx,idy] = TensorMap(randn, codomain, domain);
    end

    # function return
    return finitePEPS

end