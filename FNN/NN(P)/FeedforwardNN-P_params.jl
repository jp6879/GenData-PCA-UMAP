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

#-----------------------------------------------------------------------------------------

function P(lcm, σ)
    return ( exp( -(log(lc) - log(lcm))^2 / (2σ^2) ) ) / (lc*σ*sqrt(2π))
end

#------------------------------------------------------------------------------------------

# Leemos los datos a los que les realizamos PCA

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)

#datasignals = Matrix(df_datasignals)

# df_dataprobd = CSV.read(path_read * "\\df_PCA_Probd_60var.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

#------------------------------------------------------------------------------------------
# Funciones de pre procesamiento para escalar los datos

function MaxMin(data)
    # Calculate the minimum and maximum values for each dimension
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)

    # Scale the data to the range of 0 to 1

    # Scale the data to the range of -1 to 1
    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)

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

function curve_points(x,y)
    theta = atan.(y, x)
    r = sqrt.(x .^ 2 .+ y.^ 2)
    return r, theta
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

folds = 1
step_valid = 10
num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:2])')

σ_valid = Float32.(df_datasignals[1:step_valid:num_datos,3])
lcm_valid = Float32.(df_datasignals[1:step_valid:num_datos,4])

σ_col = Float32.(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),3])
lcm_col = Float32.(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4])

dataparams = hcat(σ_col, lcm_col)'
dataparams_valid = hcat(σ_valid, lcm_valid)'


#------------------------------------------------------------------------------------------
# Definimos la red neuronal

model = Chain(
    Dense(2, 32, selu),
    Dense(32, 64, selu),
    Dense(64, 128, selu),
    Dense(128, 64, tanh_fast),
    Dense(64, 32, tanh_fast),
    Dense(32, 2, identity),
)

# Función de loss
function loss(x,y)
    # penalty = sum(pen_l2, Flux.params(model))
    return Flux.mse(P(model(x)), P(y))
end

# Definimos el optimizador
opt = ADAM(1e-5)

# Definimos el número de épocas
epochs = 500

# Definimos el batch size
batch_size = 50

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((dataparams, datasignals), batchsize = batch_size, shuffle = false)
data_valid = Flux.DataLoader((dataparams_valid, datasignals_valid), batchsize = batch_size, shuffle = true)

# Definimos el vector donde guardamos la pérdida
losses = zeros(epochs)
losses_valid = zeros(epochs)

# Definimos el vector donde guardamos los parámetros de la red neuronal
params = Flux.params(model)

# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
epoch_iter = 0
cb = function()
    global epoch_iter
    global iter += 1
    if iter % length(data) == 0
        epoch_iter += 1
        actual_loss = loss(data.data[1], data.data[2])
        actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
        if epoch_iter % 1 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
        end
        losses[epoch_iter] = actual_loss
        losses_valid[epoch_iter] = actual_valid_loss
    end
end;

# Entrenamos la red neuronal
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb = cb)
    if epoch == 200
        opt = ADAM(1e-6)
    end
end

#------------------------------------------------------------------------------------------

# Graficamos la pérdida

pl_loss = plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(P)\\Plots\\Loss_P2.png")

#------------------------------------------------------------------------------------------


# Grafiquemos las predicciones de la red para las señales
predictions = model(dataparams)
predictions_valid = model(dataparams_valid)

predictions[1,:]
predictions[2,:]
R2_valid = R2_score(predictions, datasignals)
RMSE_valid = RMSE(predictions, datasignals)
MAE_valid = MAE(predictions, datasignals)


df_predict = DataFrame(
    pc1 = predictions[1, :],
    pc2 = predictions[2, :],
    σ = σ_col,
    lcm = lcm_col,
)

df_predict_valid = DataFrame(
    pc1 = predictions_valid[1, :],
    pc2 = predictions_valid[2, :],
    σ = σ_valid,
    lcm = lcm_valid,
)

plot_lcms_P_pred = @df df_predict StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos entrenamiento PCA de S(t)",
)

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(P)\\Plots\\Predicciones_S.png")

plot_lcms_P_pred_valid = @df df_predict_valid StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos validación PCA de S(t)",
)

savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(P)\\Plots\\Predicciones_S_valid.png")