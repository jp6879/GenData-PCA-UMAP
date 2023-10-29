using Flux
using Statistics
using Flux: train!
using Plots
using Distributions
using ProgressMeter
using MultivariateStats
using DataFrames
using CSV
using StatsPlots
using LaTeXStrings
using LinearAlgebra
using CUDA

# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################

# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 600
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 9

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lcs = Float32.(collect(range(l0, lf, length = N)))
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

function Pln(σ, lcm)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2)) / (lc * σ * sqrt(2π))) for lc in lcs]
end


# Leemos los datos a los que les realizamos PCA
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"
df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)

datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:2])')

σ_valid = df_datasignals[1:step_valid:num_datos,3]
lcm_valid = df_datasignals[1:step_valid:num_datos,4]

σ_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),3]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4]

dataparams = hcat(σ_col, lcm_col)'
dataparams_valid = hcat(σ_valid, lcm_valid)'

datasignals = datasignals[:,1:100:end]
dataparams = dataparams[:,1:100:end]

data = Flux.DataLoader((datasignals, dataparams), shuffle = false)

# model = Chain(
#     Dense(2, 15, softplus),
#     Dense(15, 50, softplus),
#     Dense(50, 150, softplus),
#     Dense(150, 100, softplus),
#     Dense(100, 50, softplus),
#     Dense(50, 32, softplus),
#     Dense(32, 15, softplus),    
#     Dense(15, 2, softplus)
# )

model = Chain(
    Dense(2, 10, softplus),
    Dense(10, 10, softplus),
    Dense(10, 2, softplus),
)


function loss(x,y)
    y_hat = model(x)
    Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
    Pln_real = Pln.(y[1,:], y[2,:])
    return mean(abs2, mean.(abs2,Pln_real .- Pln_predicted))
end

opt = ADAM(1e-4)

iter = 0
cb = function()
    global iter += 1
    actual_loss = loss(data.data[1], data.data[2])
    actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
    println("Epoch $iter || Loss = $actual_loss")# || Valid Loss = $actual_valid_loss")
    #losses[iter] = actual_loss
    #losses_valid[iter] = actual_valid_loss
end;

for i in 1:100
    Flux.train!(loss, Flux.params(model), data, opt)
end

actual_loss = loss(data.data[1], data.data[2])
actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
println("Epoch $iter || Loss = $actual_loss")

Y_hat = model(datasignals)
dataparams

P_true = Pln(data.data[2][1], data.data[2][2])
P_hat = Pln(Y_hat[1],Y_hat[2])

scatter(lcs, P_true, label = "P_true", xlabel = L"l_c", ylabel = "P(l_c)", title = "Distribución de tamaños de correlación", legend = :topleft)
scatter!(lcs, P_hat, label = "P_hat")