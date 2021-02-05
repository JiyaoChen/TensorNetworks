
using Arpack
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using Plots
using Printf
using TensorOperations

# clearconsole()

include("iDMRG.jl")

# precompile(iDMRG,(Int64,Int64))
# @time iDMRG(10,100)
numSteps = convert(Int64,10);
bondDim = convert(Int64,10);
MPS,groundStateEnergy = iDMRG(bondDim,numSteps)

# pyplot()
# plt = plot(groundStateEnergy[1:end,1],groundStateEnergy[1:end,2],line = :solid,color = :black,linewidth = 1,marker = :circle,label = "groundStateEnergy");
# gui()
# display(plt)
