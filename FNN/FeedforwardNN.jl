using Flux
using Statistics
using Flux: train!
using Plots
using Distributions
using CUDA
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
datasignals = Matrix(df_datasignals)

df_dataprobd = CSV.read(path_read * "\\df_PCA_Probd.csv", DataFrame)
dataprobd = Matrix(df_dataprobd)

# De los datos de distribuciones de probabilidad nos quedamos solo con las 2 primeras componentes principales aunque estemos perdiendo información

# Guardemos el 10% de los datos para validación

# Select from the dataframe the not divisible by 10 columns
datasignals = df_datasignals[if i%10 != 0 for i in 1:55100 end, :]
dataprobd = df_dataprobd[if i%10 != 0 for i in 1:55100 end, :]

datasignals = Matrix(datasignals)'
dataprobd = Matrix(dataprobd)'

# Select from the dataframe the divisible by 10 columns
datasignals_valid = df_datasignals[if i%10 == 0 for i in 1:55100 end, :]
dataprobd_valid = df_dataprobd[if i%10 == 0 for i in 1:55100 end, :]

datasignals_valid = Matrix(datasignals_valid)'
dataprobd_valid = Matrix(dataprobd_valid)'



#------------------------------------------------------------------------------------------ 

# Guardamos el 10% de los datos para validación

datasignals_valid = datasignals[j if j%10 == 0 for j in 1:55100, 1:2]
dataprobd_valid = dataprobd[j if j%10 == 0 for j in 1:55100, 1:2]

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
labels = false,  # Use the modified labels
title = "PCA para S(t)",
)

# Graficamos los datos de salida

plot_lcms_P = @df df_dataprobd StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para P(lc)",
)

#------------------------------------------------------------------------------------------

# Red neuronal para relacionar los datos de las señales con los de las distribuciones de probabilidad

# Definimos la red neuronal

model = Chain(
    Dense(2, 10, relu),
    Dense(10, 25, relu),
    Dense(25,10, tanh_fast),
    Dense(10, 2)
)

# Definimos la función de pérdida

loss(x, y) = Flux.mse(model(x), y)

# Definimos el optimizador

opt = ADAM(1e-6)

# Definimos el número de épocas

epochs = 10000

# Definimos el batch size

batch_size = 100

# Definimos el número de batches

n_batches = Int(ceil(size(dataprobd, 2) / batch_size))

# Usamos dataloader para cargar los datos

data = Flux.DataLoader((datasignals, dataprobd), batchsize = batch_size, shuffle = true)

# Definimos el vector donde guardamos la pérdida

losses = zeros(epochs)

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
        if epoch_iter % 1 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss")
        end
        losses[epoch_iter] = actual_loss
    end
end;

# Entrenamos la red neuronal
for _ in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb = cb)
end

# Graficamos la pérdida

plot(1:epochs, losses, xlabel = "Epochs", ylabel = "Loss", title = "Loss vs Epochs")

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales

predicteddp = model(datasignals)

df_predict = DataFrame(
    pc1 = predicteddp[1, :],
    pc2 = predicteddp[2, :],
    σs = column_σs,
    lcm = column_lcm,
)


plot_lcms_P = @df df_predict StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para P(lc)",
)


@df df_PCA_Probd StatsPlots.scatter!(
    :pc1,
    :pc2,
    marker = (0.01,5),
    labels = false,
)


# Probemos en little data acá teniamos

# Leemos los datos a los que les realizamos PCA
path_pract = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"

df_data_pract_s = CSV.read(path_pract * "\\df_PCA_Signals.csv", DataFrame)
df_data_pract_pd = CSV.read(path_pract * "\\df_PCA_Probd.csv", DataFrame)

data_pract_s = Matrix(df_data_pract_s)[:, 1:2]'
data_pract_pd = Matrix(df_data_pract_pd)[:, 1:2]'

column_σs_ld = Matrix(df_data_pract_pd)[:, 3]
column_lcm_ld = Matrix(df_data_pract_pd)[:, 4]


# Grafiquemos las predicciones de la red para las señales

predict_pract_pd = model(data_pract_s)

df_predict_pract = DataFrame(
    pc1 = predict_pract_pd[1, :],
    pc2 = predict_pract_pd[2, :],
    σs = column_σs_ld,
    lcm = column_lcm_ld,
)

plot_lcms_P_pract = @df df_predict_pract StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para P(lc)",
)

plotpd = @df df_data_pract_pd StatsPlots.scatter(
    :pc1,
    :pc2,
    marker = (0.1,5),
    labels = false,
)

plotsSs = @df df_data_pract_s StatsPlots.scatter(
    :pc1,
    :pc2,
    marker = (1,5),
    labels = false,
)