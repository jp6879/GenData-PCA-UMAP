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
#datasignals = Matrix(df_datasignals)

df_dataprobd = CSV.read(path_read * "\\df_PCA_Probd_60var.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

# df_dataprobd
# df_datasignals

# Algo que podemos hacer es invertir las direcciones de una de las componentes principales para 
# tener así datos mas parecidos a los de las entradas
# Convert the second row (row 2) to an array and multiply by -1
# row_array = Array(df_dataprobd[18, :])
# row_array .*= -1

# # Update the DataFrame with the modified row
# df_dataprobd[18, :] .= row_array
#------------------------------------------------------------------------------------------

num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:10:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:10:num_datos),1:2])')

dataprobd_valid = Float32.(Matrix(df_dataprobd[:,1:10:num_datos]))
dataprobd = Float32.(Matrix(df_dataprobd[:,setdiff(1:num_datos, 1:10:num_datos)]))

# Para 2 pcs
# dataprobd_valid = Float32.(Matrix(df_dataprobd[1:10:num_datos,1:2])')
# dataprobd = Float32.(Matrix(df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),1:2])')

σ_valid = df_datasignals[1:10:num_datos,3]
lcm_valid = df_datasignals[1:10:num_datos,4]
σ_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),3]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),4]


function MaxMin(data)
    # Calculate the minimum and maximum values for each dimension
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)

    # Scale the data to the range of -1 to 1
    scaled_data = -1 .+ 2 * (data .- min_vals) ./ (max_vals .- min_vals)

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


# scaled_datasignals = MaxMin(datasignals)
# scaled_datasignals_valid = MaxMin(datasignals_valid)

# scaled_dataprobd = MaxMin(dataprobd)
# scaled_dataprobd_valid = MaxMin(dataprobd_valid)

standarized_datasignals = Standarize(datasignals)
standarized_datasignals_valid = Standarize(datasignals_valid)

standarized_dataprobd = Standarize(dataprobd)
standarized_dataprobd_valid = Standarize(dataprobd_valid)

#------------------------------------------------------------------------------------------

# Identificación de los datos reducidos con señales y distribuciones de probabilidad originales

dim1 = dimlcm = length(lcms)
dim2 = dimσ = length(σs)

column_σs = df_datasignals[:,3]
column_lcm = df_datasignals[:,4]

#------------------------------------------------------------------------------------------

# Graficamos los datos de entrada

# plot_lcms_S = @df df_datasignals StatsPlots.scatter(
# :pc1,
# :pc2,
# group = :lcm,
# marker = (0.4,5),
# xaxis = (title = "PC1"),
# yaxis = (title = "PC2"),
# xlabel = "PC1",
# ylabel = "PC2",
# labels = false,
# title = "PCA para S(t)",
# )

# Graficamos los datos de salida, 
# plot_lcms_P = @df df_dataprobd StatsPlots.scatter(
#     :x1,
#     :x2,
#     group = :lcm,
#     marker = (0.5,5),
#     xaxis = (title = "PC1"),
#     yaxis = (title = "PC2"),
#     xlabel = "PC1",
#     ylabel = "PC2",
#     labels = false,
#     title = "PCA para P(lc)",
# )

#------------------------------------------------------------------------------------------

# Red neuronal para relacionar los datos de las señales con los de las distribuciones de probabilidad

# Definimos la red neuronal

model = Chain(
    Dense(2, 20, relu),
    Dense(20, 40, relu),
    Dense(40, 11),
)

# Definimos la función de pérdida
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)
function loss(x,y)
    penalty = sum(pen_l1, Flux.params(model))
    return Flux.mse(model(x), y) + 5e-9 * penalty
end

#loss(x, y) = Flux.mse(model(x), y)

# Definimos el optimizador

opt = ADAM(1e-6)

# Definimos el número de épocas

epochs = 100

# Definimos el batch size

batch_size = 50

# Usamos dataloader para cargar los datos

data = Flux.DataLoader((datasignals, dataprobd), batchsize = batch_size, shuffle = true)
data_valid = Flux.DataLoader((datasignals_valid, dataprobd_valid), batchsize = batch_size, shuffle = true)

# Definimos el vector donde guardamos la pérdida

losses = zeros(epochs)
losses_valid = zeros(epochs)

# Definimos el vector donde guardamos los parámetros de la red neuronal

params = Flux.params(model)

# Definimos una funcion de callback para ver el progreso del entrenamiento

iter = 0
epoch_iter = 0
cb = function()
    global iter
    global epoch_iter
    iter += 1
    # Record Loss
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
    if epoch == 50
        opt = ADAM(1e-8)
    end
end

# Graficamos la pérdida

pl_loss = plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
yaxis!(pl_loss, (0.0001, 0.5), log = true)

savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Loss_60var.png")

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales

predicteddp = model(datasignals)
predicteddp_valid = model(datasignals_valid)

# Veamos la MAE de los datos predichos por la red

function MAE(predicted, real)
    return sum(abs.(predicted .- real)) / length(predicted)
end

# Usemos el R2 score

function R2_score(predicted, real)
    return 1 - sum((predicted .- real).^2) / sum((real .- mean(real)).^2)
end

MAE_train = MAE(predicteddp, dataprobd)
MAE_valid = MAE(predicteddp_valid, dataprobd_valid)

R2_train = R2_score(predicteddp, dataprobd)
R2_valid = R2_score(predicteddp_valid, dataprobd_valid)

df_predict = DataFrame(
    pc1 = predicteddp[1, :],
    pc2 = predicteddp[2, :],
    σ = σ_col,
    lcm = lcm_col,
)

df_predict_valid = DataFrame(
    pc1 = predicteddp_valid[1, :],
    pc2 = predicteddp_valid[2, :],
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
    title = "Predicción datos entrenamiento PCA para P(lc)",
)

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Predict60var.png")


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
    title = "Predicción datos validación PCA para P(lc)",
)


savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Predict-Valid60var.png")

# @df df_PCA_Probd StatsPlots.scatter!(
#     :pc1,
#     :pc2,
#     marker = (0.01,5),
#     labels = false,
# )
