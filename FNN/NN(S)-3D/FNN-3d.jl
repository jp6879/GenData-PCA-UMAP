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
using PlotlyJS

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
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA\\PCA3D"
df_datasignals = CSV.read(path_read * "\\df_PCA_Signal_100var.csv", DataFrame)
#datasignals = Matrix(df_datasignals)

df_dataprobd = CSV.read(path_read * "\\df_PCA_Probd_86var.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

# Algo que podemos hacer es invertir las direcciones de una de las componentes principales para 
# tener así datos mas parecidos a los de las entradas

# df_dataprobd[:,2] = -df_dataprobd[:,2]



#------------------------------------------------------------------------------------------

num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:10:num_datos,1:3])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:10:num_datos),1:3])')

dataprobd_valid = Float32.(Matrix(df_dataprobd[1:10:num_datos,1:2])')
dataprobd = Float32.(Matrix(df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),1:2])')

σ_valid = df_datasignals[1:10:num_datos,4]
lcm_valid = df_datasignals[1:10:num_datos,5]
σ_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),4]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),5]

# Quedemonos con los datos de las señales y las distribuciones de probabilidad que tengan σ < 0.5

# datasignals = datasignals[:, σ_col .< 0.5]
# datasignals_valid = datasignals_valid[:, σ_valid .< 0.5]
# dataprobd = dataprobd[:, σ_col .< 0.5]
# dataprobd_valid = dataprobd_valid[:, σ_valid .< 0.5]

Plots.plot(datasignals[1, :], datasignals[2, :], seriestype = :scatter, label = "Datos de entrenamiento", xlabel = "PC1", ylabel = "PC2")
Plots.plot(dataprobd[1, :], dataprobd[2, :], seriestype = :scatter, label = "Datos de entrenamiento", xlabel = "PC1", ylabel = "PC2")

#datasignals[2, :] = -datasignals[2, :]
#datasignals_valid[2, :] = -datasignals_valid[2, :]
# dataprobd[2, :] = -dataprobd[2, :]
# dataprobd_valid[2, :] = -dataprobd_valid[2, :]


Plots.plot(datasignals[1, :], datasignals[2, :], seriestype = :scatter, label = "Datos de entrenamiento", xlabel = "PC1", ylabel = "PC2")
Plots.plot(dataprobd[1, :], dataprobd[2, :], seriestype = :scatter, label = "Datos de entrenamiento", xlabel = "PC1", ylabel = "PC2")


#------------------------------------------------------------------------------------------

# Funciones de pre procesamiento para escalar los datos

function MaxMin(data)
    # Calculate the minimum and maximum values for each dimension
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)

    # Scale the data to the range of 0 to 1
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

# Normalizamos todos los datos

# for i in 1:3
#     datasignals[i, :] = MaxMin(datasignals[i, :])
#     datasignals_valid[i, :] = MaxMin(datasignals_valid[i, :])
#     dataprobd[i, :] = MaxMin(dataprobd[i, :])
#     dataprobd_valid[i, :] = MaxMin(dataprobd_valid[i, :])
# end

#------------------------------------------------------------------------------------------

# Graficamos los datos de entrada

plot_lcms_S = @df df_datasignals StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.4,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "PCA para S(t)",
)

# Graficamos los datos de salida, 
plot_lcms_P = @df df_dataprobd StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "PCA para P(lc)",
)


df_train_P = DataFrame(
    pc1 = dataprobd[1, :],
    pc2 = dataprobd[2, :],
    σs = σ_col,
    lcm = lcm_col,
)

# Graficamos los datos de salida, 
plot_lcms_P = @df df_train_P StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :σs,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "PCA para P(lc)",
)

# Graficos interactivos

PlotlyJS.plot(
    df_datasignals, Layout(margin=attr(l=0, r=0, b=0, t=0)),
    x=:pc1, y=:pc2, z=:pc3, color=:lcm,
    type="scatter3d", mode="markers", hoverinfo="text", hovertext=:σs,title = "PCA para S(t)",
)

PlotlyJS.plot(
    df_dataprobd, Layout(margin=attr(l=0, r=0, b=0, t=0)),
    x=:pc1, y=:pc2, z=:pc3, color=:σs,
    type="scatter3d", mode="markers", hoverinfo="text", hovertext=:lcm, title = "PCA para P(lc)",
)

#------------------------------------------------------------------------------------------

# Red neuronal para relacionar los datos de las señales con los de las distribuciones de probabilidad

# Definimos la red neuronal

model = Chain(
    Dense(3, 16, selu),
    Dense(16, 32, selu),
    Dense(32, 64, leakyrelu),
    Dense(64, 128, leakyrelu),
    Dense(128, 128, leakyrelu),
    Dense(128, 64, leakyrelu),
    Dense(64, 32, selu),
    Dense(32, 16, selu),
    Dense(16, 8, selu),
    Dense(8, 2),
)

# model = Chain(
#     Dense(2, 10, relu),
#     Dense(10, 25, relu),
#     Dense(25, 50, tanh_fast),
#     Dense(50, 50, tanh_fast),
#     Dense(50, 2)
# )


# Definimos la función de pérdida

function loss(x,y)
    return Flux.Losses.mae(model(x), y)
end

#loss(x, y) = Flux.mse(model(x), y)

# Definimos el optimizador

opt = ADAM(1e-6)

# Definimos el número de épocas

epochs = 500

# Definimos el batch size

batch_size = 100

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
        if epoch_iter % 10 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
        end
        losses[epoch_iter] = actual_loss
        losses_valid[epoch_iter] = actual_valid_loss
    end
end;

# Entrenamos la red neuronal
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb = cb)
end

# Graficamos la pérdida

pl_loss = Plots.plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
Plots.plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
Plots.yaxis!(pl_loss, (-0.1, 0.6), log = true)
Plots.xlims!(100, epochs)

savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)-3D\\Plots\\Loss.png")

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales

predicteddp = model(datasignals)
predicteddp_valid = model(datasignals_valid)

#------------------------------------------------------------------------------------------
R2_score(predicteddp[1, :], dataprobd[1, :])
R2_score(predicteddp[2, :], dataprobd[2, :])

R2_score(predicteddp_valid[1, :], dataprobd_valid[1, :])
R2_score(predicteddp_valid[2, :], dataprobd_valid[2, :])

df_predict = DataFrame(
    pc1 = predicteddp[1, :],
    pc2 = predicteddp[2, :],
    pc3 = df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),"pc3"],
    σs = σ_col,
    lcm = lcm_col,
)

PlotlyJS.plot(
    df_predict, Layout(margin=attr(l=0, r=0, b=0, t=0)),
    x=:pc1, y=:pc2, z=:pc3, color=:σs,
    type="scatter3d", mode="markers", hoverinfo="text", hovertext=:lcm, title = "PCA para P(lc)",
)

df_predict_valid = DataFrame(
    pc1 = predicteddp_valid[1, :],
    pc2 = predicteddp_valid[2, :],
    σs = σ_valid,
    lcm = lcm_valid,
)

plot_lcms_P = @df df_train_P StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :σs,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "PCA para P(lc)",
)


plot_lcms_P_pred = @df df_predict StatsPlots.scatter!(
    :pc1,
    :pc2,
    group = :σs,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos entrenamiento PCA para P(lc)",
)

Plots.savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)-3D\\Plots\\Predict.png")


plot_lcms_P_pred_valid = @df df_predict_valid StatsPlots.scatter!(
    :pc1,
    :pc2,
    group = :σs,
    marker = (1,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos entrenamiento y validacion PCA para P(lc)",
)


Plots.savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)-3D\\Plots\\Predict-Valid.png")

PlotlyJS.plot(
    df_predict, Layout(margin=attr(l=0, r=0, b=0, t=0)),
    x=:pc1, y=:pc2, z=:pc3, color=:σs,
    type="scatter3d", mode="markers", hoverinfo="text", hovertext=:lcm, title = "PCA para P(lc)",
)

# @df df_PCA_Probd StatsPlots.scatter!(
#     :pc1,
#     :pc2,
#     marker = (0.01,5),
#     labels = false,
# )
