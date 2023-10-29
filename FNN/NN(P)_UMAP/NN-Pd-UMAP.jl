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
path_read_umap = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_UMAP"
df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)
#datasignals = Matrix(df_datasignals)

df_dataprobd = CSV.read(path_read_umap * "\\df_UMAP_Probd_mdist_0.05_nn_100_nc_2.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

# Algo que podemos hacer es invertir las direcciones de una de las componentes principales para 
# tener así datos mas parecidos a los de las entradas
df_datasignals[:,2] = -df_datasignals[:,2]



#------------------------------------------------------------------------------------------

num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:10:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:10:num_datos),1:2])')

dataprobd_valid = Float32.(Matrix(df_dataprobd[1:10:num_datos,1:2])')
dataprobd = Float32.(Matrix(df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),1:2])')

σ_valid = df_datasignals[1:10:num_datos,3]
lcm_valid = df_datasignals[1:10:num_datos,4]
σ_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),3]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),4]

#------------------------------------------------------------------------------------------

# Funciones de pre procesamiento para escalar los datos

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


# for i in 1:2
#     dataprobd[i,:] = Float32.(MaxMin(dataprobd[i,:]))
#     dataprobd_valid[i,:] = Float32.(MaxMin(dataprobd_valid[i,:]))
# end


#------------------------------------------------------------------------------------------

# Identificación de los datos reducidos con señales y distribuciones de probabilidad originales

dim1 = dimlcm = length(lcms)
dim2 = dimσ = length(σs)

column_σs = df_datasignals[:,3]
column_lcm = df_datasignals[:,4]

#------------------------------------------------------------------------------------------

# Graficamos los datos de entrenamiento de entrada
plot_lcms_S = @df df_datasignals StatsPlots.scatter(
:pc1,
:pc2,
group = :lcm,
marker = (1,5),
xaxis = (title = "PC1"),
yaxis = (title = "PC2"),
xlabel = "PC1",
ylabel = "PC2",
labels = false,
title = "PCA para S(t)",
)

# Graficamos los datos de entrenamiento salida 
plot_lcms_P = @df df_dataprobd StatsPlots.scatter(
    :proyX,
    :proyY,
    group = :lcm,
    marker = (0.4,5),
    xaxis = (title = "proyX"),
    yaxis = (title = "proyY"),
    xlabel = "proyX",
    ylabel = "proyY",
    labels = false,
    title = "UMAP para P(lc)",
)


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
# Red neuronal para relacionar los datos de las señales con los de las distribuciones de probabilidad

# Definimos la red neuronal

model = Chain(
    Dense(2, 32, tanh_fast),
    Dense(32, 64, tanh_fast),
    Dense(64, 128, selu),
    Dense(128, 64, selu),
    Dense(64, 32, selu),
    Dense(32, 2, identity),
)
# model = Chain(
#     Dense(2, 25, selu),
#     Dense(25, 50, selu),
#     Dense(50, 25, selu),
#     Dense(25, 15, selu),
#     Dense(15, 5, selu),
#     Dense(5, 2, identity),
# )

# Definimos la función de pérdida

# function loss(x,y)
#     return Flux.huber_loss(model(x), y)
# end

loss(x, y) = Flux.mse(model(x), y)

# Definimos el optimizador

opt = ADAM(1e-4)

# Definimos el número de épocas
epochs = 1000

# Definimos el batch size
batch_size = 100

# Normalizamos los datos
# n_datasignals = Float32.(zeros(size(datasignals)))
# n_datasignals_valid = Float32.(zeros(size(datasignals_valid)))
# n_dataprobd = Float32.(zeros(size(dataprobd)))
# n_dataprobd_valid = Float32.(zeros(size(dataprobd_valid)))

# for i in 1:2
#     n_datasignals[i,:] = Float32.(MaxMin(datasignals[i,:]))
#     n_datasignals_valid[i,:] = Float32.(MaxMin(datasignals_valid[i,:]))
#     n_dataprobd[i,:] = Float32.(MaxMin(dataprobd[i,:]))
#     n_dataprobd_valid[i,:] = Float32.(MaxMin(dataprobd_valid[i,:]))
# end

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((dataprobd, datasignals), batchsize = batch_size, shuffle = true)
data_valid = Flux.DataLoader((dataprobd_valid, datasignals_valid), batchsize = batch_size, shuffle = true)

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
    # if epoch == 100
    #     opt = ADAM(1e-5)
    # end
    if epoch == 200   # then change to use η = 0.01 for the rest.
        opt = ADAM(1e-7)
    end
end

println("R2 score: $(R2_score(model(dataprobd), datasignals)) || RMSE: $(RMSE(model(dataprobd), datasignals)) || MAE: $(MAE(model(dataprobd), datasignals))")
println("R2 score valid: $(R2_score(model(dataprobd_valid), datasignals_valid)) || RMSE valid: $(RMSE(model(dataprobd_valid), datasignals_valid)) || MAE valid: $(MAE(model(dataprobd_valid), datasignals_valid))")


# Graficamos la pérdida
pl_loss = plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
yaxis!(pl_loss, (0.00001, 0.0001), log = true)
xlims!(1, epochs)
savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(P)_UMAP\\Plots\\Loss.png")

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales

predictedsignals = model(dataprobd)
predictedsignals_valid = model(dataprobd_valid)

df_predict = DataFrame(
    pc1 = predictedsignals[1, :],
    pc2 = predictedsignals[2, :],
    σ = σ_col,
    lcm = lcm_col,
)

df_predict_valid = DataFrame(
    pc1 = predictedsignals_valid[1, :],
    pc2 = predictedsignals_valid[2, :],
    σ = σ_valid,
    lcm = lcm_valid,
)

plot_lcms_P_pred = @df df_predict StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos entrenamiento PCA para S(t)",
)

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(P)_UMAP\\Plots\\Predict.png")

plot_lcms_P_pred_valid = @df df_predict_valid StatsPlots.scatter!(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,
    title = "Predicción datos PCA para P(lc)",
)

savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Predict-Valid.png")

# @df df_PCA_Probd StatsPlots.scatter!(
#     :pc1,
#     :pc2,
#     marker = (0.01,5),
#     labels = false,
# )
