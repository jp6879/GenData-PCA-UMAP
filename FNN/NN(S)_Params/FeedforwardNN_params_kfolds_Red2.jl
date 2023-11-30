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

#------------------------------------------------------------------------------------------

# Distribucion de probabilidad log-normal

function Pln(lcm::Float32, σ::Float32)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

#------------------------------------------------------------------------------------------

# Leemos los datos a los que les realizamos PCA de las señales

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"

df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)

#------------------------------------------------------------------------------------------
# Funciones de pre procesamiento para escalar los datos

# Normalización Max-Min
function MaxMin(data)
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)
    return scaled_data

end

# Estandarización
function Standarize(data)
    mean_vals = mean(data, dims=1)
    std_devs = std(data, dims=1)
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

# Realetive Root Mean Squared Error
function RRMSE(predicted, real)
    return sqrt(mean((predicted .- real).^2)) / mean(real)
end

# Relative Mean Absolute Error
function RMAE(predicted, real)
    return mean(abs.(predicted .- real)) / mean(real)
end

# Mean Absolute Percentaje Error
function MAPE(predicted, real)
    return mean(abs.((predicted .- real) ./ real))
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

# Guardamos los datos de validacion de cada NN en cada fold
out_of_sample_data = []
out_of_sample_pred = []

# Guardamos los datos de la funcion loss de cada NN en cada fold
loss_folds = []
loss_folds_valids = []

# Guardamos las metricas de validacion de cada NN en cada fold
scores = []

for k in 1:folds

    # Usamos 5 conjuntos disjuntos de datos de validación del 10% total de los datos para cada fold

    datasignals_valid = Float32.(Matrix(df_datasignals[k^2:step_valid:num_datos,1:3])')
    datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),1:3])')

    σ_valid = df_datasignals[k^2:step_valid:num_datos,4]
    lcm_valid = df_datasignals[k^2:step_valid:num_datos,5]
    
    σ_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),4]
    lcm_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),5]
    
    dataparams = hcat(lcm_col, σ_col)'
    dataparams_valid = hcat(lcm_valid, σ_valid)'

    # Definimos la red neuronal
    model = Chain(
        Dense(3, 5, relu),
        Dense(5, 15, relu),
        Dense(15, 50, relu),
        Dense(50, 200, relu),
        Dropout(0.5),
        Dense(200, 100, relu),
        Dense(100, 64, relu),
        Dense(64, 10, relu),
        Dense(10, 5, relu),    
        Dense(5, 2, softplus)
    )

    # Función de loss
    function loss(x,y)
        return Flux.mse(model(x), y)
    end

    # Definimos el metodo de aprendizaje y la tasa de aprendizaje
    η = 1e-4
    opt = ADAM(η)

    # Definimos el número de épocas
    epochs = 2500

    # Definimos el tamaño del batch
    batch_size = 1101

    # Usamos dataloader para cargar los datos
    data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true)
    data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

    # Definimos el vector donde guardamos la pérdida
    losses = []
    losses_valid = []

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

    # Entrenamos la red neuronal con el loss mse dividiendo en 5 la tasa de aprendizaje cada 500 epocas
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
        if epoch % 1000 == 0
            η /= 5
            opt = ADAM(η)
        end
    end

    # Predicción de la red en la validacion
    predictions_valid = model(datasignals_valid)

    # Métricas de validación de la red
    R2_valid = R2_score(predictions_valid, dataparams_valid)
    RMSE_valid = RMSE(predictions_valid, dataparams_valid)
    MAE_valid = MAE(predictions_valid, dataparams_valid)

    actual_scores = [R2_valid, RMSE_valid, MAE_valid]

    push!(scores, actual_scores)

    # Guardamos los datos de validación y las predicciones de la red
    push!(out_of_sample_data, dataparams_valid)
    push!(out_of_sample_pred, predictions_valid)

    println("Fold $k terminado con score de validación R2 = $R2_valid RMSE = $RMSE_valid y MAE = $MAE_valid")

end

R2_scores = [scores[i][1] for i in 1:length(scores)]
RMSE_scores = [scores[i][2] for i in 1:length(scores)]
MAE_scores = [scores[i][3] for i in 1:length(scores)]


plot_scores = #Plots.scatter(collect(1:5),R2_scores, label = "R2", xlabel = "Folds", ylabel = "Valor", title = "Metricas validación para cada fold", legend = :best)
Plots.scatter(collect(1:5),RMSE_scores, label = "RMSE", xlabel = "Fold", ylabel = "Valor", title = "Metricas validación para cada fold",tickfontsize=12, labelfontsize=14, legendfontsize=12, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
Plots.scatter!(collect(1:5),MAE_scores, label = "MAE")
plot_scores = Plots.plot(plot_scores, legend = :right, size = (600,400))

Plots.savefig(plot_scores,"C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Red2_Plots_final_relu\\Metricas_K-Folds_red2.pdf")

# Al haber testeado en todos estos datos y seguir obteniendo valores similares podemos asegurar que el modelo no está sobreajustando
# Ahora podemos re entrenar el modelo tranquilamente

num_datos = Int(size(df_datasignals, 1))

datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:3])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:3])')

σ_valid = df_datasignals[1:step_valid:num_datos,4]
lcm_valid = df_datasignals[1:step_valid:num_datos,5]

σ_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),5]

dataparams = hcat(lcm_col, σ_col)'
dataparams_valid = hcat(lcm_valid, σ_valid)'

model = Chain(
    Dense(3, 5, relu),
    Dense(5, 15, relu),
    Dense(15, 50, relu),
    Dense(50, 200, relu),
    Dropout(0.5),
    Dense(200, 100, relu),
    Dense(100, 64, relu),
    Dense(64, 10, relu),
    Dense(10, 5, relu),    
    Dense(5, 2, softplus)
)

# Función de loss
function loss(model,x,y)
    y_hat = model(x)
    return Flux.mse(y_hat, y)
end


# y_hat = model(data.data[1])
# y_hat[1,:]

# Pln_predictions = [zeros(Float32, N) |> Flux.gpu for _ in 1:length(y_hat[1,:])] |> Flux.gpu
# @time Pln_predictions = [Pln(y_hat[1, i], y_hat[2, i]) for i in 1:length(y_hat[1,:])]

# y = model(data.data[1]) |> Flux.cpu
# @time Pln.(y[1,:], y[2,:])

# Loss compuesto
function composed_loss(x,y)
    y_hat = model(x)
    Pln_predicted = Pln.(y_hat[1,:] |> Flux.cpu, y_hat[2,:] |> Flux.cpu) |> Flux.gpu
    Pln_real = Pln.(y[1,:] |> Flux.cpu, y[2,:] |> Flux.cpu) |> Flux.gpu
    return mean(Flux.mse.(Pln_predicted,Pln_real)) + Flux.mse(y_hat, y)
end


# Definimos el batch size
batch_size = 1101

# datasignals = CuArray(datasignals)
# dataparams = CuArray(dataparams)
# datasignals_valid = CuArray(datasignals_valid)
# dataparams_valid = CuArray(dataparams_valid)

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true) 
data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

# Definimos el vector donde guardamos la pérdida
losses = []
losses_valid = []

# Parámetros de la red neuronal
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


# Definimos el número de épocas
epochs = 1000

# opt_state = Flux.setup(Flux.Adam(0.01), model)

# @time for epoch in 1:epochs
#     for (x, y) in data
#         grads = gradient(model -> loss(model, x, y), model)
#         Flux.update!(opt_state, model, grads[1])
#     end
# end

# Definimos el modo de aprendizaje y la tasa de aprendizaje
η = 1e-9
opt = ADAM(η)

# Entrenamos la red neuronal con el loss mse
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
end

# Plots de la función de loss
pl_loss = Plots.plot(losses[1000:end], xlabel = "Épocas", ylabel = "Loss", label = "Datos de entrenamiento",tickfontsize=12, labelfontsize=14, legendfontsize=12, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, size = (600,400))
Plots.plot!(losses_valid[1000:end], xlabel = "Épocas", ylabel = "Loss", label = "Datos de validación")
Plots.savefig(pl_loss,"C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\FNN\\NN(S)_Params\\Red2_Plots_final_relu\\Loss_Red2.pdf")

# Predicciones de la red
predictions = model(datasignals)
predictions_valid = model(datasignals_valid)

#------------------------------------------------------------------------------------------
# Medidas de error globales

R2_train = R2_score(predictions, dataparams)
RMSE_train = RMSE(predictions, dataparams)
MAE_train = MAE(predictions, dataparams)
RMAE_train = RMAE(predictions, dataparams)

R2_valid = R2_score(predictions_valid, dataparams_valid)
RMSE_valid = RMSE(predictions_valid, dataparams_valid)
MAE_valid = MAE(predictions_valid, dataparams_valid)
RMAE_train = RMAE(predictions_valid, dataparams_valid)

println("R2 train = $R2_train || RMSE train = $RMSE_train || MAE train = $MAE_train")
println("R2 valid = $R2_valid || RMSE valid = $RMSE_valid || MAE valid = $MAE_valid")

#------------------------------------------------------------------------------------------
# Medidas de error puntuales
N = length(predictions[1,:])
N_valid = length(predictions_valid[1,:])

RMSE_scores = zeros(N)
MAE_scores = zeros(N)
RMAE_scores = zeros(N)

RMSE_scores_valid = zeros(N_valid)
MAE_scores_valid = zeros(N_valid)
RMAE_scores_valid = zeros(N_valid)

for i in 1:N
    RMSE_scores[i] = RMSE(predictions[:,i], dataparams[:,i])
    MAE_scores[i] = MAE(predictions[:,i], dataparams[:,i])
    RMAE_scores[i] = RMAE(predictions[:,i], dataparams[:,i])
end

for i in 1:N_valid
    RMSE_scores_valid[i] = RMSE(predictions_valid[:,i], dataparams_valid[:,i])
    MAE_scores_valid[i] = MAE(predictions_valid[:,i], dataparams_valid[:,i])
    RMAE_scores_valid[i] = RMAE(predictions_valid[:,i], dataparams_valid[:,i])
end

#------------------------------------------------------------------------------------------

# Plots de los errores de la predicion de la red
params_error = PlotlyJS.scatter(
    x = datasignals_valid[1,:],
    y = datasignals_valid[2,:],
    mode = "markers",
    hoverinfo = "text",
    hovertext = MAE_scores_valid,
    marker = attr(
        color = MAE_scores_valid,  # Use the color_vector for color mapping
        colorscale = "Hot",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "MAE",
    ),
    labels = true,
)

params_layout = Layout(
    title = "Error absoluto en la predicion de los parámetros datos de validacion",
    xaxis = attr(title = "PC1"),
    yaxis = attr(title = "PC2"),
)

params_plot = PlotlyJS.plot([params_error], params_layout)

RMSE_PCA = PlotlyJS.scatter(
    x = datasignals_valid[1,:],
    y = datasignals_valid[2,:],
    mode = "markers",
    hoverinfo = "text",
    hovertext = RMSE_scores_valid,
    marker = attr(
        color = RMSE_scores_valid,  # Use the color_vector for color mapping
        colorscale = "Hot",  # Choose a predefined colormap (e.g., "Viridis")
        colorbar_title = "RMSE",
    ),
    labels = true,
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
)

RMSE_plot = PlotlyJS.plot([RMSE_PCA], MAE_layout)
