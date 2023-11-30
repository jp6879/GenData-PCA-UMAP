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
N =  500
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.05
lf = 10

lcs = Float32.(collect(range(l0, lf, length = N)))


# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.005:6
σs = 0.01:0.01:1

function Pln(σ, lcm)
    # P = zeros(length(lcs))
    
    # for i in 1:length(lcs)
    #     P[i] = ( exp( -(log(lcs[i]) - log(lcm))^2 / (2σ^2) ) ) / (lcs[i]*σ*sqrt(2π))
    # end

    # return P

    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

#------------------------------------------------------------------------------------------

# Leemos los datos a los que les realizamos PCA

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"

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

    # Scale the data to the range of 0.01 to 1

    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)

    # Scale the data to the range of -1 to 1
    #scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)

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

folds = 1
step_valid = 10
num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:2])')

σ_valid = df_datasignals[1:step_valid:num_datos,3]
lcm_valid = df_datasignals[1:step_valid:num_datos,4]

σ_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),3]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4]

dataparams = hcat(lcm_col, σ_col)'
dataparams_valid = hcat(lcm_valid, σ_valid)'

# datasignals[1,:] = -datasignals[1,:]
# datasignals_valid[1,:] = -datasignals_valid[1,:]
#datasignals[2,:] = -datasignals[2,:]
#datasignals_valid[2,:] = -datasignals_valid[2,:]

# for i in 1:2
#     dataparams[i,:] = Float32.(MaxMin(dataparams[i,:]))
#     datasignals[i,:] = Float32.(MaxMin(datasignals[i,:]))
#     dataparams_valid[i,:] = Float32.(MaxMin(dataparams_valid[i,:]))
#     datasignals_valid[i,:] = Float32.(MaxMin(datasignals_valid[i,:]))
# end


scatter(dataparams[1,:], dataparams[2,:], xlabel = "lcm", ylabel = "σ", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))
scatter!(datasignals[1,:], datasignals[2,:], xlabel = "lcm", ylabel = "σ", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

# for i in 1:2
#     dataparams[i,:] = Float32.(MaxMin(dataparams[i,:]))
#     dataparams_valid[i,:] = Float32.(MaxMin(dataparams_valid[i,:]))
# end

out_of_sample_data = []
out_of_sample_pred = []

loss_plots = []
scores = []

fold = 0
#------------------------------------------------------------------------------------------
# Test
# Extract from the data all the values with σ>0.6

# datasignals_test = df_datasignals[df_datasignals[!, 3] .< 0.6, :]

# datasignals = Float32.(Matrix(datasignals_test[:,1:2])')
# σ_col = Float32.(datasignals_test[:,3])
# lcm_col = Float32.(datasignals_test[:,4])
# dataparams = Float32.(hcat(σ_col, lcm_col)')

# for i in 1:2
#     dataparams[i,:] = Float32.(MaxMin(dataparams[i,:]))
#     datasignals[i,:] = Float32.(MaxMin(datasignals[i,:]))
#     dataparams_valid[i,:] = Float32.(MaxMin(dataparams_valid[i,:]))
#     datasignals_valid[i,:] = Float32.(MaxMin(datasignals_valid[i,:]))
# end




#scatter(dataparams[1,:], dataparams[2,:], xlabel = "σ", ylabel = "lcm", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

# Definimos la red neuronal

model = Chain(
    Dense(2, 5, relu),
    Dense(5, 15, relu),
    Dense(15, 50, relu),
    BatchNorm(50),
    Dense(50, 200, relu),
    Dropout(0.2),
    Dense(200, 100, relu),
    BatchNorm(100),
    Dense(100, 64, relu),
    Dense(64, 50, relu),
    Dense(50, 32, relu),
    BatchNorm(32),
    Dense(32, 10, relu),
    Dense(10, 5, relu),    
    Dense(5, 2, softplus)
)

# model = Chain(
#     Dense(2, 25, relu),
#     Dense(25, 32, relu),
#     Dense(32, 25, relu),
#     Dense(25, 2, softplus),
# )


# data.data[2]
# model(data.data[1])
#plot(lcs, Pln_pred[1])

# Función de loss
function composed_loss(x,y)
    y_hat = model(x)
    # Pln_predicted = softmax.(Pln.(y_hat[1,:], y_hat[2,:]))
    # Pln_predicted = softmax.(Pln.(y_hat[1,:], y_hat[2,:]))
    # Pln_real = softmax.(Pln.(y[1,:], y[2,:]))

    # return mean(abs2, mean.(abs2, Pln_predicted .- Pln_real)) + Flux.mae(y_hat, y)
    # return mean(abs2,Flux.kldivergence.(Pln_predicted, Pln_real)) #+ Flux.mse(y_hat, y)
    # return mean(abs2,Flux.crossentropy.(Pln_predicted, Pln_real)) + Flux.mse(y_hat, y)

    # return sum(Flux.logitcrossentropy.(Pln_predicted, Pln_real))
    return Flux.mae(y_hat, y)

    # return Flux.mae(y_hat, y)

end

# Definimos el optimizador
opt = ADAM(1e-8)

# Definimos el número de épocas
epochs = 1000

# Definimos el batch size
batch_size = 1101

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true)
data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

# sigma_pred, lcm_pred = model(data.data[1][:,1:2])
# true_sigma, true_lcm = data.data[2][:,1:2]

# y_hat = model(data.data[1])

# data.data[2]

# y_hat[1,:]
# y_hat[2,:]

# Pln.(y_hat[1,:], y_hat[2,:])

# model(data.data[1][:,1:2])[:,1]

# Pln_true = Pln.(data.data[2][1,1:2], data.data[2][2,1:2])
# Pln_predict = Pln.(y_hat[1,1:2], y_hat[1,1:2])

# plot(lcs, softmax(Pln_true[1]))
# plot!(lcs, softmax(Pln_predict[1]))

# sum(CrossEntropy(softmax(softmax(Pln_true[1])), softmax(softmax(Pln_predict[1]))))
# sum(Flux.logitcrossentropy.(Pln_predict, softmax.(Pln_true)))

# Definimos el vector donde guardamos la pérdida
losses = zeros(epochs)
losses_valid = zeros(epochs)

# Definimos el vector donde guardamos los parámetros de la red neuronal
params = Flux.params(model)

# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
cb = function()
    global iter += 1
    if iter % length(data)*10 == 0
        actual_loss = composed_loss(data.data[1], data.data[2])
        actual_valid_loss = composed_loss(data_valid.data[1], data_valid.data[2])
        println("Iter $iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
    end
end;

# Entrenamos la red neuronal
for epoch in 1:epochs
    Flux.train!(composed_loss, Flux.params(model, opt), data, opt, cb = cb)
end

actual_loss = composed_loss(data.data[1], data.data[2])
actual_valid_loss = composed_loss(data_valid.data[1], data_valid.data[2])
println("Iter $iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")

# Graficamos la pérdida

#pl_loss = plot(1:epochs, losses, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
#plot!(1:epochs, losses_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)

# Métricas de validación de la red

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales
predictions = model(datasignals)
predictions_valid = model(datasignals_valid)

predictions[1,:]
predictions[2,:]
R2_train = R2_score(predictions, dataparams)
RMSE_train = RMSE(predictions, dataparams)
MAE_train = MAE(predictions, dataparams)

R2_valid = R2_score(predictions_valid, dataparams_valid)
RMSE_valid = RMSE(predictions_valid, dataparams_valid)
MAE_valid = MAE(predictions_valid, dataparams_valid)

println("R2 train = $R2_train || RMSE train = $RMSE_train || MAE train = $MAE_train")
println("R2 valid = $R2_valid || RMSE valid = $RMSE_valid || MAE valid = $MAE_valid")


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

scatter(dataparams[1,:], dataparams[2,:], xlabel = "lcm", ylabel = "σ", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

plot_lcms_P_pred = @df df_predict StatsPlots.scatter!(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "lcm"),
    yaxis = (title = "σ"),
    xlabel = "lcm",
    ylabel = "σ",
    labels = false,
    title = "Predicción datos entrenamiento",
)

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Plots\\Predicción_params.png")



# σ_recover = predictions[1,:] .* cos.(predictions[2,:])
# lcm_recover = predictions[1,:] .* sin.(predictions[2,:])

# scatter(σ_recover, lcm_recover, xlabel = "σ", ylabel = "lcm", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

plot_lcms_P_pred_valid = @df df_predict_valid StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (1,5),
    xaxis = (title = "σ"),
    yaxis = (title = "lcm"),
    xlabel = "σ",
    ylabel = "lcm",
    labels = false,
    title = "Predicción datos validacion",
)

savefig(plot_lcms_P_pred_valid, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Plots\\Predicción_params_valid.png")
