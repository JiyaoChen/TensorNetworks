include("header.jl")
include("pepsUnitCell.jl")

Lx = 2;
Ly = 1;
unitCellLayout = [1 2 ; 2 1];

chiB = 2;
chiE = 3;
truncBelowE = 1e-6;
d = 2;

latticeTens = Array{Array{Float64, 5}, 2}(undef, Lx, Ly);
for idx = 1 : Lx, idy = 1 : Ly
    latticeTens[idx, idy] = randn(chiB, chiB, d, chiB, chiB);
end
# latticeTens[1, 1] = randn(3, 5, d, 4, 6));
# latticeTens[2, 1] = randn(4, 6, d, 3, 5));
iPEPS = pepsUnitCell(Lx, Ly, latticeTens, unitCellLayout);

# test getindex
for lx = 1:2*Lx, ly = 1:2*Ly
    @info lx, ly, iPEPS[lx, ly]
end