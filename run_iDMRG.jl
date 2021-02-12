push!(LOAD_PATH, pwd())
using iDMRG

# clear console
Base.run(`clear`)
χ = 64
tol = 1e-16
@time iDMRG.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
@time iDMRG.iDMRG2(χ=χ, maxNumSteps=100, setSym="Z2", tol=tol)
# @time iDMRG.iDMRG2(χ=χ, maxNumSteps=6, setSym="")