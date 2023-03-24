using CSV, LinearAlgebra, DataFrames, Tables, QuadGK, GenericSchur, Plots, Interpolations, TimerOutputs
tmr = TimerOutput();

tobs = 898*24*3600
F0 = 13.22
F1 = 1.498
F2 = -0.00167
F3 = 4.119
logemin = log10(290)
logemax = log10(10^4)

function phi(x)
    return 10^(-F0 - (F1*log10(x))/(1 + F2 * abs(log10(x))^F3))
end

function Afuncalt(x)
    y = log10(x)
    return 10^(3.57 + 2.007*y -0.5263* y^2 +0.0922 * y^3 -0.0072* y^4)
end

#calculating the RHS Matrix
function RHSmatrix(energy_nodes,dxs_array)
    NumNodes = length(energy_nodes)
    DeltaE = diff(log.(energy_nodes))
    RHS = zeros(BigFloat, NumNodes, NumNodes)
    for i in 1:NumNodes
        for j in i+1:NumNodes
            RHS[i,j] = DeltaE[j-1] * dxs_array[j,i] * energy_nodes[j]^(-1) * energy_nodes[i]^2
        end
    end
    return RHS
end

function phi_0_calc(energy_nodes,NumNodes)
    # phi_0 = zeros(BigFloat, NumNodes)
    return [10^(-F0-(F1*log10(energy_nodes[i]))/(1 + F2 * abs(log10(energy_nodes[i]))^F3)) * (energy_nodes[i])^2 for i = 1:NumNodes]
end

function sigma_calc(energy_nodes,NumNodes,a,b)
    # phi_0 = zeros(BigFloat, NumNodes)
    return Diagonal([(a/b)*(1 - 1/(2*b*energy_nodes[i]) * log(1 + 2*b* energy_nodes[i])) for i = 1:NumNodes])
end

function dxs_calc(energy_nodes,NumNodes,a,b)
    dxs_array = Array{BigFloat, 2}(undef, NumNodes, NumNodes)
    for i in 1:NumNodes
        for j in 1:NumNodes
            f(x) = a*(energy_nodes[i]/x)* 1/((1 + 2*b*(x-energy_nodes[i]))^2)
            dxs_array[i,j] = quadgk(f, energy_nodes[i], 10^4)[1]
        end
    end
    return dxs_array
end

function phisol_calc_2(v, ci, w, energy_nodes, E)
    energy_nodes = convert(Array{BigFloat}, energy_nodes)
    phisol = (v * ci) .* exp.(w) .* energy_nodes.^(-2)
    phisol = convert(Array{Float64}, phisol)
    energy_nodes = convert(Array{Float64}, energy_nodes)
    phisol_interp = linear_interpolation(energy_nodes, phisol,extrapolation_bc=Line()) 
    return phisol_interp(E)
end

function phisol_calc(sigma_array,dxs_array,energy_nodes,NumNodes,phi_0,E)
    w, v = eigen(Matrix(-sigma_array + RHSmatrix(energy_nodes,dxs_array)))
    @timeit tmr "ci" ci = (v \ phi_0)
    # @timeit tmr "phisol_calc_2" phisol_calc_2(v, ci, w, energy_nodes, E)
    convert(Array{BigFloat}, energy_nodes)
    phisol = (v * ci) .* exp.(w) .* energy_nodes.^(-2)
    convert(Array{Float64}, phisol)
    convert(Array{Float64}, energy_nodes)
    phisol_interp = linear_interpolation(energy_nodes, phisol,extrapolation_bc=Line()) 
    return phisol_interp(E)
end

#Eigen calculator
function eigcalc(E::Float64,num::Int64,a::Float64,b::Float64)
    NumNodes = num
    energy_nodes = 10 .^(range(logemin,stop=logemax,length=NumNodes))

    #flux across the energy spectrum
    @timeit tmr "phi" phi_0 = phi_0_calc(energy_nodes,NumNodes)
    # phi_0 = zeros(BigFloat, NumNodes)
    # for i in 1:NumNodes
    #     phi_0[i]=10^(-F0-(F1*log10(energy_nodes[i]))/(1 + F2 * abs(log10(energy_nodes[i]))^F3)) * (energy_nodes[i])^2
    # end

    #cross section array intialization
    @timeit tmr "sigma" sigma_array = sigma_calc(energy_nodes,NumNodes,a,b)
    # sigma_array  =  zeros(BigFloat, NumNodes, NumNodes)
    # for i in 1:NumNodes
    #     sigma_array[i,i] = (a/b)*(1 - 1/(2*b*energy_nodes[i]) * log(1 + 2*b* energy_nodes[i])) 
    # end

    #differential cross section intialization
    @timeit tmr "dxs" dxs_array = dxs_calc(energy_nodes,NumNodes,a,b)
    # dxs_array = Array{BigFloat, 2}(undef, NumNodes, NumNodes)
    # for i in 1:NumNodes
    #     for j in 1:NumNodes
    #         f(x) = a*(energy_nodes[i]/x)* 1/((1 + 2*b*(x-energy_nodes[i]))^2)
    #         dxs_array[i,j] = quadgk(f, energy_nodes[i], 10^4)[1]
    #     end
    # end

    # RHS = RHSmatrix(energy_nodes,dxs_array)
    # sum = Matrix(-sigma_array + RHS)
    # w, v = eigen(sum)
    # ci = (v \ phi_0)
    # energy_nodes = convert(Array{BigFloat}, energy_nodes)
    # ci = convert(Array{Float64}, ci)
    # v  = convert(Array{Float64}, v)
    # w  = convert(Array{Float64}, w)
    # phisol = (v * ci) .* exp.(w) .* energy_nodes.^(-2)
    @timeit tmr "phisol" return phisol_calc(sigma_array,dxs_array,energy_nodes,NumNodes,phi_0,E)
    # phisol = convert(Array{Float64}, phisol)
    # energy_nodes = convert(Array{Float64}, energy_nodes)
    # phisol_interp = linear_interpolation(energy_nodes, phisol,extrapolation_bc=Line()) 
    # return phisol_interp(E)
end

#Integral
function events(N,Neig)
    Aval = 10 .^(range(-2.5,stop=-0.5,length=N))
    Bval = 10 .^(range(-6,stop=-1,length=N))
    dat_fin = zeros(Float64, N*N, 3)

    steps = 1000
    deltaE = (10.0^4 - 290.0)/steps
    enn = range(290.0,stop=10.0^4,length=steps)

    # dat_fin = zeros(Float64, N, 2)
    print(sum([eigcalc(enn[i],Neig,10^-2.5,10^-1)* Afuncalt(enn[i]) for i = 1:steps])* deltaE *tobs,"\n")
    # for i in 1:N
    #     dat_fin[(i-1)N+1:i*N,1] .=  Aval[i]
    #     for j in 1:N
    #         dat_fin[(i-1)*N+j,2] = Bval[j]
    #         # total_sum = [eigcalc(enn[i],Neig,dat_fin[(i-1)*N+j,1],dat_fin[(i-1)*N+j,2])* Afuncalt(enn[i]) for i = 1:steps]
    #         # dat_fin[i*N+j,3] = sum([eigcalc(enn[i],Neig,dat_fin[i*N+j, 1],dat_fin[i*N+j, 2])* Afuncalt(enn[i]) for i = 1:steps])*deltaE *tobs
    #         dat_fin[(i-1)*N+j,3] = sum([eigcalc(enn[k],Neig,Aval[i],Bval[j])* Afuncalt(enn[k]) for k = 1:steps])* deltaE *tobs
    #         print(i,j,"\n")
    #     end
    # end
    # CSV.write("AvsB_contour.csv",  Tables.table(dat_fin), writeheader=false)
    # plotting(dat_fin)
    # total_sum = [eigcalc(enn[i],Neig,0.000,0.001)* Afuncalt(enn[i]) for i = 1:steps]
    # print(sum(total_sum)* deltaE *tobs)
    print("Thy bidding is done, My Master" )
end

function plotting(arr)
    # Plot the scatter plot and join the points
    # plot(arr[:,1], arr[:,2], xscale=:log10, yscale=:log10, xlabel="E", ylabel="Phi")
    contour(arr[:,1], arr[:,2], arr[:,3], levels=[0.1], fill=true, color=:viridis, legend=:none)
end

print("Enter N: ")
N = parse(Int64, readline()) 
print("Enter Neig: ")
Neig = parse(Int, readline())

@time events(N,Neig)



