using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
using Flux.Data: DataLoader
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
N = 1700
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.05
lf = 10

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.005:6
σs = 0.01:0.01:1

#------------------------------------------------------------------------------------------
# Leemos los datos a los que les realizamos PCA

# Leemos los datos a los que les realizamos PCA
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA\\PCAXL"
df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)
#datasignals = Matrix(df_datasignals)

df_dataprobd = CSV.read(path_read * "\\df_PCA_Probd_86var.csv", DataFrame)
#dataprobd = Matrix(df_dataprobd)

#------------------------------------------------------------------------------------------

num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals = Float32.(Matrix(df_datasignals[:,1:3])')
dataprobd = Float32.(Matrix(df_dataprobd[:,1:3])')

σ_col = df_datasignals[:,4]
lcm_col = df_datasignals[:,5]

# # dataprobd_valid = Float32.(Matrix(df_dataprobd[:,1:10:num_datos]))
# # dataprobd = Float32.(Matrix(df_dataprobd[:,setdiff(1:num_datos, 1:10:num_datos)]))

# # Para 2 pcs
# dataprobd_valid = Float32.(Matrix(df_dataprobd[1:10:num_datos,1:2])')
# dataprobd = Float32.(Matrix(df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),1:2])')


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

for i in 1:3
    datasignals[i, :] = MaxMin(datasignals[i, :])
    # datasignals_valid[i, :] = MaxMin(datasignals_valid[i, :])
    dataprobd[i, :] = MaxMin(dataprobd[i, :])
    # dataprobd_valid[i, :] = MaxMin(dataprobd_valid[i, :])
end

batch_size = 1101

data = Flux.DataLoader((datasignals |> Flux.gpu, dataprobd |> Flux.gpu), batchsize = batch_size, shuffle = true)

Plots.scatter(data.data[1][1,:] |> Flux.cpu, data.data[1][2,:] |> Flux.cpu)
Plots.scatter(data.data[2][1,:] |> Flux.cpu, data.data[2][2,:] |> Flux.cpu)

# data_valid = Flux.DataLoader((n_datasignals_valid |> Flux.gpu, n_dataprobd_valid |> Flux.gpu), batchsize = batch_size, shuffle = true)

# function random_point_in_sphere(dim, min_radius, max_radius)
#     distance = (max_radius - min_radius) .* (rand(Float32,1) .^ (1f0 / dim)) .+ min_radius
#     direction = randn(Float32,dim)
#     unit_direction = direction ./ norm(direction)
#     return distance .* unit_direction
# end



# function concentric_sphere(dim, inner_radius_range, outer_radius_range,
#                            num_samples_inner, num_samples_outer; batch_size = 64)
#     data = []
#     labels = []
#     for _ in 1:num_samples_inner
#         push!(data, reshape(random_point_in_sphere(dim, inner_radius_range...), :, 1))
#         push!(labels, ones(1, 1))
#     end
#     for _ in 1:num_samples_outer
#         push!(data, reshape(random_point_in_sphere(dim, outer_radius_range...), :, 1))
#         push!(labels, -ones(1, 1))
#     end
#     data = cat(data..., dims=2)
#     labels = cat(labels..., dims=2)
#     DataLoader((data |> Flux.gpu, labels |> Flux.gpu); batchsize=batch_size, shuffle=true,
#                       partial=false)
# end

diffeqarray_to_array(x) = reshape(Flux.gpu(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    
    # node = NeuralODE(Flux.Chain(
    #     Dense(input_dim, 10, relu),
    #     Dense(10, 25, relu),
    #     Dense(25, 50, tanh_fast),
    #     Dense(50, 50, tanh_fast),
    #     Dense(50, input_dim) |> Flux.gpu ), (0.f0, 1f0), Tsit5(), save_everystep = false,
    #     reltol = 1f-2, abstol = 1f-2, save_start = false) |> Flux.gpu

    node = NeuralODE(Flux.Chain(Flux.Dense(input_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, input_dim)) |> Flux.gpu,
                     (0.f0, 6.f0), Tsit5(), save_everystep = false,
                     reltol = 1f-3, abstol = 1f-3, save_start = false) |> Flux.gpu

    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)

    return Flux.Chain((x, p=node.p) -> node(x, p),
                 Array,
                 diffeqarray_to_array,
                 Flux.Dense(input_dim, out_dim, selu) |> Flux.gpu), node.p |> Flux.gpu
end

# function plot_contour(model, npoints = 300)
#     grid_points = zeros(Float32, 2, npoints ^ 2)
#     idx = 1
#     x = range(-4f0, 4f0, length = npoints)
#     y = range(-4f0, 4f0, length = npoints)
#     for x1 in x, x2 in y
#         grid_points[:, idx] .= [x1, x2]
#         idx += 1
#     end
#     sol = reshape(model(grid_points |> Flux.gpu), npoints, npoints) |> Flux.cpu

#     return contour(x, y, sol, fill = true, linewidth=0.0)
# end

# scatter(data.data[1][1,:],data.data[1][2,:])
# scatter(data.data[2][1,:], data.data[2][2,:])

loss_node(x, y) = mean((model(x) .- y) .^ 2)

# loss_node(x,y) = Flux.huber_loss(model(x), y)

# println("Generating Dataset")

# dataloader = concentric_sphere(2, (0f0, 2f0), (3f0, 4f0), 2000, 2000; batch_size = 256)

# dataloader.data[1]
# dataloader.data[2]

iter = 0
# epoch_iter = 0
# cb = function()
#     global iter
#     global epoch_iter
#     iter += 1
#     # Record Loss
#     if iter % length(data) == 0
#         epoch_iter += 1
#         actual_loss = loss_node(data.data[1], data.data[2])
#         actual_valid_loss = loss_node(data_valid.data[1], data_valid.data[2])
#         if epoch_iter % 1 == 0
#             println("Epoch $epoch_iter || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
#         end
#         # losses[epoch_iter] = actual_loss
#         # losses_valid[epoch_iter] = actual_valid_loss
#     end
# end;

cb = function()
    global iter
    iter += 1
    if iter % length(data) == 0
        println("Iteration $iter || Loss = $(loss_node(data.data[1], data.data[2]))") # || Loss valid = $(loss_node(data_valid.data[1], data_valid.data[2]))")
    end
end

model, parameters = construct_model(3, 3, 25, 10)

opt = Adam(1e-3)

for _ in 1:50
    Flux.train!(loss_node, Flux.params(parameters, model), data, opt, cb = cb)
end

predicted = model(data.data[1] |> Flux.gpu) |> Flux.cpu
#predicted_valid = model(datasignals_valid |> Flux.gpu) |> Flux.cpu

Plots.scatter(predicted[1,:], predicted[2,:])
#scatter!(predicted_valid[1,:], predicted_valid[2,:])