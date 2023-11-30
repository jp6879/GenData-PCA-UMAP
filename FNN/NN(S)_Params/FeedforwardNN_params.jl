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
using PlotlyJS
using CUDA
using Random


# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################
# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 2000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 50

lcs = Float32.(collect(range(l0, lf, length = N)))

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1



function Pln(lcm, σ)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end


#------------------------------------------------------------------------------------------

# Leemos los datos a los que les realizamos PCA

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"

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
# Utilizamos la técnica k-fold de validación cruzada para prevenir el overfitting de la red neuronal
# Definimos el número de folds
folds = 5
step_valid = 10
num_datos = Int(size(df_datasignals, 1))

datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:3])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:3])')

σ_valid = df_datasignals[1:step_valid:num_datos,4]
lcm_valid = df_datasignals[1:step_valid:num_datos,5]

σ_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),5]

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


Plots.scatter(dataparams[1,:], dataparams[2,:], xlabel = "lcm", ylabel = "σ", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))
Plots.scatter(datasignals[1,:], datasignals[2,:], datasignals[3,:], xlabel = "PC1", ylabel = "PC2", zlabel = "PC3", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

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

# Definimos la red neuronal

# model = Chain(
#     Dense(3, 5, relu),
#     Dense(5, 15, relu),
#     Dense(15, 50, relu),
#     Dense(50, 200, relu),
#     Dropout(0.3),
#     Dense(200, 100, relu),
#     Dense(100, 64, relu),
#     Dense(64, 10, relu),
#     Dense(10, 5, relu),    
#     Dense(5, 2, softplus)
# )

model = Chain(
    Dense(3, 25, relu),
    Dense(25, 32, relu),
    Dense(32, 25, relu),
    Dense(25, 2, softplus),
)

# Función de loss
function loss(x,y)
    y_hat = model(x)
    return Flux.mse(y_hat, y)
end

function composed_loss(x,y)
    y_hat = model(x)
    Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
    Pln_real = Pln.(y[1,:], y[2,:])
    return mean(Flux.mse.(Pln_predicted,Pln_real)) + Flux.mse(y_hat, y)
end


# y_hat = model(data.data[1])
# y = data.data[2]
# Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
# Pln_real = Pln.(y[1,:], y[2,:])

# Pln_predicted[1,:]
# Pln_real[1,:]

# scatter(lcs, Pln_predicted[1,:])
# scatter!(lcs, Pln_real[1,:])

# mean(Flux.mse.(Pln_predicted,Pln_real))

# Definimos el optimizador
opt = ADAM(1e-6)

# Definimos el batch size
batch_size = 1101

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = false)
data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = false)

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
losses = []
losses_valid = []

losses_composed = []
losses_composed_valid = []

# Definimos el vector donde guardamos los parámetros de la red neuronal
params = Flux.params(model)

# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
cb = function()
    global iter += 1
    if iter % length(data) == 0
        epoch = iter ÷ length(data)
        actual_loss = loss(data.data[1], data.data[2])
        actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
        if epoch % 10 == 0
            println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
        end
        push!(losses, actual_loss)
        push!(losses_valid, actual_valid_loss)
    end
end;

cb2 = function()
    global iter += 1
    if iter % length(data) == 0
        epoch = iter ÷ length(data)
        actual_loss = composed_loss(data.data[1], data.data[2])
        actual_valid_loss = composed_loss(data_valid.data[1], data_valid.data[2])
        if epoch % 10 == 0
            println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
        end
        push!(losses_composed, actual_loss)
        push!(losses_composed_valid, actual_valid_loss)
    end
end;

# Definimos el número de épocas
epochs = 500

# Entrenamos la red neuronal con el loss normal
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
end

epochs = 50

# Entrenamos la red neuronal con el loss compuesto para mejorar la predicción de los parámetros
for epoch in 1:epochs
    Flux.train!(composed_loss, Flux.params(model, opt), data, opt, cb=cb2)
end

actual_loss = composed_loss(data.data[1], data.data[2])
actual_valid_loss = composed_loss(data_valid.data[1], data_valid.data[2])
println("Iter $iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")

# Graficamos la pérdida

pl_loss = plot(losses[200:end], xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true, ylims = (0.005, 0.05))
plot!(losses_valid[200:end], xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)

Plots.savefig(pl_loss, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Plots\\Loss.png")

Plots.plot(losses_composed, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de entrenamiento", logy = true)
Plots.plot(losses_composed_valid, xlabel = "Epocas", ylabel = "Loss", label = "Loss datos de validación", logy = true)
# Métricas de validación de la red

#------------------------------------------------------------------------------------------

# Grafiquemos las predicciones de la red para las señales
predictions = model(datasignals)
predictions_valid = model(datasignals_valid)

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

Plots.scatter(dataparams[1,:], dataparams[2,:], xlabel = "lcm", ylabel = "σ", title = "Datos de entrenamiento", size = (800, 400))

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

#------------------------------------------------------------------------------------------
# Medidas de error puntuales
N = length(predictions[1,:])
N_valid = length(predictions_valid[1,:])

R2_scores = zeros(N)
RMSE_scores = zeros(N)
MAE_scores = zeros(N)

R2_scores_valid = zeros(N_valid)
RMSE_scores_valid = zeros(N_valid)
MAE_scores_valid = zeros(N_valid)

MAE(predictions[:,1], dataparams[:,1])
RMSE(predictions[:,1], dataparams[:,1])

R2_score(predictions[:,1], dataparams[:,1])

for i in 1:N
    R2_scores[i] = R2_score(predictions[:,i], dataparams[:,i])
    RMSE_scores[i] = RMSE(predictions[:,i], dataparams[:,i])
    MAE_scores[i] = MAE(predictions[:,i], dataparams[:,i])
end

for i in 1:N_valid
    R2_scores_valid[i] = R2_score(predictions_valid[:,i], dataparams_valid[:,i])
    RMSE_scores_valid[i] = RMSE(predictions_valid[:,i], dataparams_valid[:,i])
    MAE_scores_valid[i] = MAE(predictions_valid[:,i], dataparams_valid[:,i])
end

predictions[2,1]

#------------------------------------------------------------------------------------------
MAE_sigma = zeros(N)
MAE_lcm = zeros(N)
for i in 1:N
    MAE_sigma[i] = MAE(predictions[2,i], dataparams[2,i])
    MAE_lcm[i] = MAE(predictions[1,i], dataparams[1,i])
end

#------------------------------------------------------------------------------------------

# Datos reales

Real_data = PlotlyJS.scatter(
    x = dataparams[1,1:20:end],
    y = dataparams[2,1:20:end],
    mode = "markers",
    xaxis = (title = "lcm"),
    yaxis = (title = "σ"),
    xlabel = "lcm",
    ylabel = "σ",
)

Layout_true = Layout(
    scene = attr(
        xaxis_title = "lcm",
        yaxis_title = "σ",
        xaxis = (title = "lcm"),
        yaxis = (title = "σ"),
        xlabel = "lcm",
        ylabel = "σ",
    ),
    title = "Datos de entrenamiento que deberia predecir la red",
)

Real_data_plot = PlotlyJS.plot([Real_data], Layout_true)


# Create a 3D scatter plot
MAE_plot = PlotlyJS.scatter(
    x = predictions[1,1:20:end],
    y = predictions[2,1:20:end],
    mode = "markers",
    hoverinfo = "text",
    hovertext = MAE_scores[1:20:end],
    marker = attr(
        color = MAE_scores[1:20:end],  # Use the color_vector for color mapping
        colorscale = "Viridis",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "Error abs normalizado",
    ),
    labels = false,
    xaxis = (title = "lcm"),
    yaxis = (title = "σ"),
    xlabel = "lcm",
    ylabel = "σ",
)

# Create a layout
layout_MAE = Layout(
    scene = attr(
        xaxis_title = "lcm",
        yaxis_title = "σ",
    ),
    title = "Predicciones de entrenamiento con MAE",
)

# Create a PlotlyJS plot
MAE_plot_final = PlotlyJS.plot([MAE_plot], layout_MAE)

# Create a 3D scatter plot
RMSE_plot = PlotlyJS.scatter(
    x = predictions[1,1:20:end],
    y = predictions[2,1:20:end],
    mode = "markers",
    hoverinfo = "text",
    hovertext = RMSE_scores[1:20:end],
    marker = attr(
        color = RMSE_scores[1:20:end],  # Use the color_vector for color mapping
        colorscale = "Viridis",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "Raiz de error cuadrático",
    ),
    labels = false,
    xaxis = (title = "lcm"),
    yaxis = (title = "σ"),
    xlabel = "lcm",
    ylabel = "σ",
)

# Create a layout
layout_RMSE = Layout(
    scene = attr(
        xaxis_title = "lcm",
        yaxis_title = "σ",
    ),
    title = "Predicciones de entrenamiento con RMSE",
)

# Create a PlotlyJS plot
PlotlyJS.plot([RMSE_plot], layout_RMSE)


MAE_PCA = PlotlyJS.scatter(
    x = datasignals[1,1:25000],
    y = datasignals[2,1:25000],
    mode = "markers",
    hoverinfo = "text",
    hovertext = MAE_lcm[1:25000],
    marker = attr(
        color = MAE_lcm[1:25000],  # Use the color_vector for color mapping
        range_color=[0.01,0.1],
        colorscale = "Hot",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "MAE",
    ),
    labels = false,
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
)

MAE_layout2 = Layout(
    scene = attr(
        xaxis_title = "PC1",
        yaxis_title = "PC2",
    ),
    title = "Error absoluto en LCM",
)

MAE_plot_final2 = PlotlyJS.plot([MAE_PCA], MAE_layout2)

#------------------------------------------------------------------------------------------

savefig(plot_lcms_P_pred, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Plots\\Predicción_params_xl.png")

scatter(dataparams_valid[1,:], dataparams_valid[2,:], xlabel = "lcm", ylabel = "σ", label = "Datos de validación", title = "Datos de validación", size = (800, 400))

# σ_recover = predictions[1,:] .* cos.(predictions[2,:])
# lcm_recover = predictions[1,:] .* sin.(predictions[2,:])

# scatter(σ_recover, lcm_recover, xlabel = "σ", ylabel = "lcm", label = "Datos de entrenamiento", title = "Datos de entrenamiento", size = (800, 400))

plot_lcms_P_pred_valid = @df df_predict_valid StatsPlots.scatter!(
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
