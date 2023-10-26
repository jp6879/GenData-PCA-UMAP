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

# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################

# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 2000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 15

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

#------------------------------------------------------------------------------------------

# Leemos los datos a los que les realizamos PCA

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)

#------------------------------------------------------------------------------------------
# Funciones de pre procesamiento para escalar los datos

function MaxMin(data)
    # Calculate the minimum and maximum values for each dimension
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)

    # Scale the data to the range of 0 to 1

    # Scale the data to the range of -1 to 1
    scaled_data = -1 .+ 2*(data .- min_vals) ./ (max_vals .- min_vals)

    return scaled_data

end

function Standarize(data)
    # Calculate the mean and standard deviation for each dimension
    mean_vals = mean(data, dims=1)
    std_devs = std(data, dims=1)

    # Standardize the data
    standardized_data = (data .- mean_vals) ./ std_devs

    return standardized_data
end

#------------------------------------------------------------------------------------------
# Metricas de validacion de la red neuronal

# Root Mean Squared Error
function RMSE(predicted, real)
    return sqrt(sum((predicted .- real).^2) / length(predicted))
end

# Mean Absolute Error
function MAE(predicted, real)
    return sum(abs.(predicted .- real)) / length(predicted)
end

# R2 score
function R2_score(predicted, real)
    return 1 - sum((predicted .- real).^2) / sum((real .- mean(real)).^2)
end

#------------------------------------------------------------------------------------------
# Regularizaciones L1 y L2 para la red neuronal
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)

#------------------------------------------------------------------------------------------

# Test
# Extract from the data all the values with σ < 0.6

datasignals_test = df_datasignals[df_datasignals[!, 3] .< 0.6, :]

datasignals = Float32.(Matrix(datasignals_test[:,1:2])')
σ_col = datasignals_test[:,3]
lcm_col = datasignals_test[:,4]
dataparams = hcat(σ_col, lcm_col)'

#------------------------------------------------------------------------------------------

function data_maker(x, len_seq)
    N = length(x)
    X = []
    while N > 0
        if (N - len_seq) <= 0
            break
        end
        vec = [[xi] for xi in x[N - len_seq:N]] # Esto lo que crea es un vector de secuencias de tamaño len_seq 
        push!(X, vec) # Almacenamos la secuencia en X
        N -= len_seq + 1 
    end 
    # Una vez finalizado esto devolvemos los valores de X y Y que son vectores de secuencias.
    return Vector{Vector{Float32}}.(X)
end

#------------------------------------------------------------------------------------------

# Vamos a separar estos datos en secuencias de tamaño 51 para que cada uno represente una transformada de Fourier
length_sequence = 100

X = data_maker(datasignals, length_sequence)
Y = data_maker(dataparams, length_sequence)


modelRNN = Chain(
RNN(length_sequence => 10, relu, init=Flux.glorot_uniform(gain=0.01)),
Dense(10, 50, relu, init = Flux.glorot_uniform(gain=0.01)),
RNN(50 => 10, relu, init=Flux.glorot_uniform(gain=0.01)),
LSTM(10 => 5, init=Flux.glorot_uniform(gain=0.01)),
Dense(5, 2)
)

function loss(x, y, m)
    loss = sum(Flux.mse(m(xi),yi) for (xi, yi) in zip(x,y))
    return loss
end;

data = zip(X, Y)

ps = Flux.params(modelRNN)

opt = Adam(1e-4)

lossRNN = []

iter = 0
epoch_iter = 0

cb = function()
    global iter
    global epoch_iter
    iter += 1
    # Record Loss

    if iter % length(data) == 0
        epoch_iter += 1
        actual_loss = 0
        actual_loss_test = 0

        for (x, y) in data
            actual_loss += loss(x, y, modelRNN)
        end

        if epoch_iter % 100 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss")
        end

        #push!(lossRNN, actual_loss)

    end
end

for _ in 1:1000
    Flux.train!(loss, ps, data, opt, cb = cb)
end