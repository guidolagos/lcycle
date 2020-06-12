################################################################################################
# CODE DESCRIPTION:
################################################################################################
# Runs a barebones implementation of the SDDP algorithm for the multi-stage discretized LifeCycle problem
#
# Last execution:
# Date:			Jun 1, 2020
# Julia version:	1.4.0
# Ipopt.jl version:	0.6.2
# JuMP.jl version:	0.21.2
# PrettyTables version:	0.9.0
################################################################################################

################################################################################################
# PACKAGES USED:
################################################################################################

using JuMP, Ipopt, Random, PrettyTables, Test, JLD, Distributions, LatinHypercubeSampling, Printf

################################################################################################
# PARAMETERS:
################################################################################################

X₀ = 2.0
Y₀ = 2.0

r = .02
λ = .25
σ = .2

ϕ = .01
β = .1

ρ = .2

κ = .6

δ = .03

Δt = 1.	#.5
R = 30	#12
T = 50	#16

τ = 1e-5
γ = 4.0	#1.25	# γ>1 reflects risk aversion
# Note: u(x) = (τ+x)^(1-γ) / (1-γ)
ϵ = 5.0

Nsamples = 12	# number of samples per each stage
Noutsamples = 100	# 100

Nsddp = 150	# 200

Δsddp_iter_save = 50	# 200
file_name = "cuts.jld"

train_cuts = true
read_cuts = !train_cuts

impose_no_shortsale = true
impose_nonnegative_wealth = false
impose_nonnegative_terminal_wealth = false
impose_budget_feasible = false

verb_node = true

Random.seed!(1234)
t = (0.0):Δt:T
Ξψ = randn(length(t), Nsamples)
ΞY = randn(length(t), Nsamples)

#Ξ, _ = LHCoptim(Nsamples, length(t), 1000)
#Ξ = quantile.(Normal(), (Ξ' .- .5)/Nsamples)

#ξmin = 2
#ξmax = -ξmin
# CI with 68.2% confidence: [-1, 1]
#         95.4% confidence: [-2, 2]
#         99.7% confidence: [-3, 3]

θmax = 1e8
bigM = 1e5

################################################################################################

gψ(ξψ) = exp( (r + σ*λ - σ^2/2)*Δt + σ*ξψ )

gY(ξψ, ξY) = exp( (ϕ - β^2/2)*Δt + β*(ρ*ξψ + sqrt(1-ρ^2)*ξY) )

################################################################################################

cmax = X₀ * ones(length(t))
let y = zeros(length(t))	# y is local variable
	for (k, tk) in enumerate(t)
		y[k] =  if tk == first(t)
				Y₀
			elseif tk <= R
				y[k-1] * maximum( gY.(Ξψ[k, :], ΞY[k, :]) )
			elseif tk == R+1
				y[k-1] * κ
			else
				y[k-1]
			end
	end
	for (k, tk) in enumerate(t), (j, tj) in enumerate(t)
		if tj-tk >= 0
			global cmax[k] += exp(-δ*(tj-tk)) * y[j]
		else
			continue
		end
        end
end
cmax = cmax*1

#lcstring = "--------------------------------------------#"
lcstring = "#"

lsep = "------------------------------------------------------------------------------------------------\n"

################################################################################################
# Save parameters and cuts in a file:
################################################################################################
function save_parameters()
save(
	file_name,
	"Ξψ", Ξψ,
	"ΞY", ΞY,
	"cmax", cmax,
	"C", C
    )
end

################################################################################################
# Optimization problem that solves DH node of the Life-cycle problem:
################################################################################################

function solve_node(tk, X, Y, cuts_dict, ξψ, ξY, cmaxk)

if tk == last(t)
	# NOTE: utility function u(x) is as follows:
	# u(x) = (τ + x)^(1-γ) / (1-γ)		if x≥0
	#        M*x + (τ+0)^(1-γ) / (1-γ)	if x<0
	# and its derivative is:
	# u'(x) = (τ+x)^(-γ)	if x≥0
	#         M		if x<0
	if X + Y * Δt >= 0
		obj = exp(-δ*tk) * ϵ * (τ + X + Y * Δt)^(1-γ) / (1-γ) * Δt
		A = exp(-δ*tk) * ϵ * (τ + X + Y * Δt)^(-γ) * Δt
		B = 0
	else
		obj = exp(-δ*tk) * ϵ * (bigM * (X + Y * Δt) + (τ+0)^(1-γ) / (1-γ)) * Δt
		A = exp(-δ*tk) * ϵ * (bigM) * Δt
		B = 0
	end

	return 0, 0, 0, A, B, obj
end

θ₀ = cuts_dict[:θ₀]
∇Xp = cuts_dict[:∇Xp]
Xpast = cuts_dict[:Xpast]
∇Yp = cuts_dict[:∇Yp]
Ypast = cuts_dict[:Ypast]

Ncuts = length(θ₀)

model = Model(
		optimizer_with_attributes(
			Ipopt.Optimizer,
			"print_level" => 0,
			"max_iter" => 10000
			)
		)

@variables(model, begin
		0 <= c <= cmaxk
		Xp[1:Nsamples]
		θ[1:Nsamples] <= θmax
		α
		ψ
	end)

if tk <= R
	@variable(model, Yp[1:Nsamples])
else
	@variable(model, Yp)
end


if impose_no_shortsale
	@constraint(model, α >= 0)
	@constraint(model, ψ >= 0)
end

if impose_nonnegative_wealth
	@constraints(model, begin
		α * (1+r)^Δt + ψ * gψ(minimum(ξψ)) >= 0
		α * (1+r)^Δt + ψ * gψ(maximum(ξψ)) >= 0
		end)
end

if impose_nonnegative_terminal_wealth
	if tk == t[end-1]
		@constraint(model, Xp .+ Δt * Yp .>= 0)
	end
end

if impose_budget_feasible
	@constraint(model, Xp .>= -cmaxk)
end

@constraint(model, conA, X + (Y - c) * Δt == α + ψ)

if tk <= R
	@constraint(model, conB, Yp .== gY.(ξψ, ξY) * Y)
elseif tk == R+1
	@constraint(model, conB, Yp == κ * Y)
else
	@constraint(model, conB, Yp == Y)
end

@constraint(model, Xp .== α * (1+r)^Δt + ψ * gψ.(ξψ))

if tk <= R
	for i in 1:Ncuts, j in 1:Nsamples
		@constraint(model,
			θ[j] <= θ₀[i] + ∇Xp[i] * (Xp[j]-Xpast[i]) + ∇Yp[i] * (Yp[j]-Ypast[i])
			)
	end
else
	for i in 1:Ncuts, j in 1:Nsamples
		@constraint(model,
			θ[j] <= θ₀[i] + ∇Xp[i] * (Xp[j]-Xpast[i]) + ∇Yp[i] * (Yp-Ypast[i])
			)
	end
end

@NLobjective(model, Max, exp(-δ*tk) * (τ + c)^(1-γ) / (1-γ) * Δt + sum(θ[i] for i in 1:Nsamples) / Nsamples )

optimize!(model)

print("$(lcstring)\ttermination status = ")
if termination_status(model) == MOI.LOCALLY_INFEASIBLE
	printstyled(termination_status(model), "\n"; bold = true, color = :red)
elseif termination_status(model) == MOI.NORM_LIMIT
	printstyled(termination_status(model), "\n"; bold = true, color = :red)
elseif termination_status(model) == MOI.NUMERICAL_ERROR
	printstyled(termination_status(model), "\n"; bold = true, color = :red)
elseif termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
	printstyled(termination_status(model), "\n"; bold = true, color = :yellow)
elseif termination_status(model) == MOI.ITERATION_LIMIT
	printstyled(termination_status(model), "\n"; bold = true, color = :yellow)
else
	println(termination_status(model))
end

@printf("%s\tX.in:\t%+.1f\n", lcstring, X)
@printf("%s\tY.in:\t%+.1f\n", lcstring, Y)
@printf("%s\tα*:\t%+.1f\n", lcstring, value(α))
@printf("%s\tψ*:\t%+.1f\n", lcstring, value(ψ))
@printf("%s\tc*:\t%.1f\n", lcstring, value(c))
println("$lcstring\tc≈cmax?\t\t$(value(c)≈cmaxk)")
println("$lcstring\tany θ≈θmax?\t$(any(value.(θ).≈θmax))")
if abs(X + (Y - value(c)) * Δt - value(α) - value(ψ)) > 1e-4
	printstyled("$lcstring\tX+(Y-c)*Δt - (α+ψ):\t$(X + (Y - value(c)) * Δt - value(α) - value(ψ))\n"; color = :blue, bold = true)
	sleep(1)
else
	println("$lcstring\tX+(Y-c)*Δt - (α+ψ):\t$(X + (Y - value(c)) * Δt - value(α) - value(ψ))")
end
#println("$lcstring\tA*:\t$(dual(conA))")
#println("$lcstring\tB*:\t$(sum(dual.(conB)))")

if termination_status(model) == MOI.LOCALLY_INFEASIBLE
#	println(model)
#	sleep(20)
end

return value(c), value(α), value(ψ), dual(conA), dual.(conB), objective_value(model)

# Termination status' received so far from Ipot:
#ALMOST_LOCALLY_SOLVED
#LOCALLY_SOLVED
#LOCALLY_INFEASIBLE
#ITERATION_LIMIT
#NUMERICAL_ERROR
#NORM_LIMIT
end

################################################################################################
# Train SDDP algorithm (i.e. generate cuts by doing Nsddp iterations):
################################################################################################

if train_cuts

# Data structure that will contain the data of the cuts:
C = []
for tk in t
	# initialize empty dictionary of cuts
	push!(C,
		Dict(
			:θ₀	=> Float64[],
			:∇Xp	=> Float64[],
			:Xpast	=> Float64[],
			:∇Yp	=> Float64[],
			:Ypast	=> Float64[]
		)
	)
end
# C[k]: cuts for problem at time t[k]
# C[k][:θ₀][i]

times_iter = Float64[]
times_node = Float64[]

# Iterates through number of SDDP-algorithm repetitions
for n_iter in 1:Nsddp
	########################################################################################
	# FORWARD STEP — solves the current approximation and gets a proposed solution ∀ time
	########################################################################################

	Δtime = time()

	# Here X and Y are arrays storing the history of state variables
	X = zeros(length(t))
	X[1] = X₀
	Y = zeros(length(t))
	Y[1] = Y₀
	for (k, tk) in enumerate(t)

		# 0. In fwd step we don't need to solve last problem
		if tk == last(t)
			continue
		end

		println("$lsep$lcstring FWD it. $n_iter\tnode $k of $(length(t)):")
		
		# 1. Solve for current solution:
		_, α, ψ, _, _, _ =
			solve_node(tk, X[k], Y[k], C[k], Ξψ[k, :], ΞY[k, :], cmax[k])

		# 2. sample stage randomness:
		rand_ind = rand(1:Nsamples)
		ξψ_sample = Ξψ[k, rand_ind]
		ξY_sample = ΞY[k, rand_ind]

		# 3. compute and store outgoing state variables X and Y:
		X[k+1] =  α * (1+r)^Δt + ψ * gψ(ξψ_sample)
		Y[k+1] = if tk <= R
				gY(ξψ_sample, ξY_sample) * Y[k]
			elseif tk == R+1
				κ * Y[k]
			else
				Y[k]
			end
	end

	########################################################################################
	# BACKWARD STEP
	########################################################################################
	
	for (k, tk) in Iterators.reverse(enumerate(t))

		# 0. In bwd step we don't need to solve first period's problem
		if tk == first(t)
			continue
		end

		println("$lsep$lcstring BWD it. $n_iter\tnode $k of $(length(t)):")
		
		# 1. Solve problem with current cuts and gather new info on cuts:
		_, _, _, A, B, obj =
			solve_node(tk, X[k], Y[k], C[k], Ξψ[k, :], ΞY[k, :], cmax[k])

		# 2. Store new info on cuts with current dual variables:
		# θ₀:	current optimal value obj
		push!(C[k-1][:θ₀], obj)
		# ∇Xp:	optimal dual variables A (a value)
		push!(C[k-1][:∇Xp], 1*A)
		# Xpast:	current state variable X[k] (a value)
		push!(C[k-1][:Xpast], X[k])
		# ∇Yp:	optimal dual variables B (a value)
		push!(C[k-1][:∇Yp],
			if tk <= R
				Δt*A + sum( gY(Ξψ[k, j], ΞY[k, j]) * B[j] for j in 1:Nsamples )
			elseif tk == R+1
				Δt*A + κ*B
			else
				Δt*A + 1*B
			end
			)
		# Ypast:	current state variable Y[k] (a value)
		push!(C[k-1][:Ypast], Y[k])
	end

	Δtime = time()-Δtime

	push!(times_iter, Δtime)
	push!(times_node, Δtime/(length(t)-1)/2)
	println("$lsep$lcstring time of iteration\t$(times_iter[end])")
	println("$lcstring avg time per node\t$(times_node[end])$lsep")


	if n_iter % Δsddp_iter_save == 0
		print("$lcstring saving cuts... ")
		save_parameters()
		println("done.$lsep")
	end
	
end
print(lsep)

end

################################################################################################
# READ CUTS:
################################################################################################

if read_cuts
	Ξψ = load(file_name, "Ξψ")
	ΞY = load(file_name, "ΞY")
	cmax = load(file_name, "cmax")
	C = load(file_name, "C")
end

################################################################################################
# EXPERIMENTS:
################################################################################################

Ceq(ut) = ( ut*(1-γ) / length(t) )^(1/(1-γ))

UB = -Inf
LB = +Inf

# Upper bound is objective function of first node:
_, _, _, _, _, UB = solve_node(t[1], X₀, Y₀, C[1], Ξψ[1, :], ΞY[1, :], cmax[1])

CeqUB = Ceq(UB)

println("UB:\t", UB)
println("CeqUB:\t", CeqUB)
sleep(5)

# Lower bound is sample of performances:
Ξψout = randn(length(t), Noutsamples)
ΞYout = randn(length(t), Noutsamples)

u = zeros(Noutsamples)
for n_iter in 1:Noutsamples
	X = zeros(length(t))
	X[1] = X₀
	Y = zeros(length(t))
	Y[1] = Y₀
	for (k, tk) in enumerate(t)
		print(lsep)
		println("$lcstring it. $n_iter node $k of $(length(t)):")
		if tk == t[end]
			u[n_iter] += exp(-δ*tk) * ϵ * (τ + X[k] + Y[k] * Δt)^(1-γ) / (1-γ) * Δt
		else	
			c, α, ψ, _, _, _ =
				solve_node(t[k], X[k], Y[k], C[k], Ξψ[k, :], ΞY[k, :], cmax[k])

			u[n_iter] += exp(-δ*tk) * (τ + c)^(1-γ) / (1-γ) * Δt

			ξψ_sample, ξY_sample = randn(2)
		
			X[k+1] =  α * (1+r)^Δt + ψ * gψ(ξψ_sample)
			Y[k+1] = if tk <= R
					gY(ξψ_sample, ξY_sample) * Y[k]
			         elseif tk == R+1
					κ * Y[k]
			         else
					Y[k]
			         end
		end
	end
end

αconf = .01	# 1-αconf confidence interval
zstar = quantile(Normal(), 1-αconf/2)	
LB = mean(u) - zstar * std(u) / Noutsamples
CeqLB = Ceq(LB)

@printf("[LB, UB] = [ %+.4f , %+.4f ]\n", LB, UB)
@printf("[C_eq LB, C_eq UB] = [ %+.4f , %+.4f ]\n", CeqLB, CeqUB)
@printf("1-CeqLB/CeqUB: %+.8f\n", 1-CeqLB/CeqUB)

println("CeqLB = ", CeqLB)
println("CeqUB = ", CeqUB)
