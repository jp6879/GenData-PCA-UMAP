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

df_dataprobd = CSV.read(path_read_umap * "\\df_UMAP_Probd.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

# Algo que podemos hacer es invertir las direcciones de una de las componentes principales para 
# tener así datos mas parecidos a los de las entradas

# df_dataprobd[:,2] = -df_dataprobd[:,2]

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

# Normalización Z de datos


#------------------------------------------------------------------------------------------

# Identificación de los datos reducidos con señales y distribuciones de probabilidad originales

dim1 = dimlcm = length(lcms)
dim2 = dimσ = length(σs)

column_σs = df_datasignals[:,3]
column_lcm = df_datasignals[:,4]

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
    :proyX,
    :proyY,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "proyX"),
    yaxis = (title = "proyY"),
    xlabel = "proyX",
    ylabel = "proyY",
    labels = false,
    title = "UMAP para P(lc)",
)

#------------------------------------------------------------------------------------------

# Red neuronal para relacionar los datos de las señales con los de las distribuciones de probabilidad

# Definimos la red neuronal

model = Chain(
    Dense(2, 10, relu),
    Dense(10, 25, relu),
    Dense(25, 50, tanh_fast),
    Dense(50, 50, tanh_fast),
    Dense(50, 2)
)

# Definimos la función de pérdida

function loss(x,y)
    return Flux.mse(model(x), y)
end

#loss(x, y) = Flux.mse(model(x), y)

# Definimos el optimizador

opt = ADAM(1e-4)

# Definimos el número de épocas

epochs = 1000

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
    if epoch == 200   # then change to use η = 0.01 for the rest.
        opt = ADAM(1e-6)
    end
end

# Graficamos la pérdida

pl_loss = plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
yaxis!(pl_loss, (0.0001, 0.1), log = true)
xlims!(5, epochs)

savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Loss.png")

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales

predicteddp = model(datasignals)
predicteddp_valid = model(datasignals_valid)

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

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Predict.png")


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


savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\Plots\\Predict-Valid.png")

# @df df_PCA_Probd StatsPlots.scatter!(
#     :pc1,
#     :pc2,
#     marker = (0.01,5),
#     labels = false,
# )
