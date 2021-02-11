using Revise
includet("./iDMRG.jl")
# using .iDMRG
# import iDMRG

# clear console
Base.run(`clear`)
for var in names(Main)
    try
        eval(parse("$var=missing"))
    catch e
    end
end
GC.gc()

# clearconsole()

# include("iDMRG.jl")

# # precompile(iDMRG,(Int64,Int64))
# # @time iDMRG(10,100)
# numSteps = convert(Int64,10);
# bondDim = convert(Int64,10);
# MPS,groundStateEnergy = iDMRG(bondDim,numSteps)

# pyplot()
# plt = plot(groundStateEnergy[1:end,1],groundStateEnergy[1:end,2],line = :solid,color = :black,linewidth = 1,marker = :circle,label = "groundStateEnergy");
# gui()
# display(plt)

# Revise.include("iDMRG2.jl");

iDMRG2()