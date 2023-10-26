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
df_dataprobd = CSV.read(path_read_umap * "\\df_UMAP_Probd_mdist0.1_nn_100.csv", DataFrame)
#------------------------------------------------------------------------------------------

num_datos = Int(size(df_datasignals, 1)) # Numero de datos

datasignals_valid = Float32.(Matrix(df_datasignals[1:10:num_datos,1:2])')
datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:10:num_datos),1:2])')

# dataprobd_valid = Float32.(Matrix(df_dataprobd[:,1:10:num_datos]))
# dataprobd = Float32.(Matrix(df_dataprobd[:,setdiff(1:num_datos, 1:10:num_datos)]))

# Para 2 pcs
dataprobd_valid = Float32.(Matrix(df_dataprobd[1:10:num_datos,1:2])')
dataprobd = Float32.(Matrix(df_dataprobd[setdiff(1:num_datos, 1:10:num_datos),1:2])')


function Standarize(data)
    # Calculate the mean and standard deviation for each dimension
    mean_vals = mean(data, dims=1)
    std_devs = std(data, dims=1)

    # Standardize the data
    standardized_data = (data .- mean_vals) ./ std_devs

    return standardized_data
end

function MaxMin(data)
    # Calculate the minimum and maximum values for each dimension
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)

    # Scale the data to the range of -1 to 1
    scaled_data = -1 .+ 2 * (data .- min_vals) ./ (max_vals .- min_vals)

    return scaled_data

end

n_datasignals = zeros(size(datasignals))
n_datasignals_valid = zeros(size(datasignals_valid))
n_dataprobd = zeros(size(dataprobd))
n_dataprobd_valid = zeros(size(dataprobd_valid))

for i in 1:2
    n_datasignals[i,:] = MaxMin(datasignals[i,:])
    n_datasignals_valid[i,:] = MaxMin(datasignals_valid[i,:])
    n_dataprobd[i,:] = MaxMin(dataprobd[i,:])
    n_dataprobd_valid[i,:] = MaxMin(dataprobd_valid[i,:])
end

# Plot valid signals and valid probd

scatter(datasignals_valid[1,:], datasignals_valid[2,:], xlabel = "PC1", ylabel= "PC2", tittle = "S(t) validacion")
scatter(dataprobd_valid[1,:], dataprobd_valid[2,:], xlabel = "x", ylabel="y", tittle = "P(lc) validacion")

# standarized_dataprobd = Standarize(dataprobd)
# standarized_dataprobd_valid = Standarize(dataprobd_valid)


σ_valid = df_datasignals[1:10:num_datos,3]
lcm_valid = df_datasignals[1:10:num_datos,4]
σ_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),3]
lcm_col = df_datasignals[setdiff(1:num_datos, 1:10:num_datos),4]

batch_size = 100

data = Flux.DataLoader((n_datasignals |> Flux.gpu, n_dataprobd |> Flux.gpu), batchsize = batch_size, shuffle = true)
data_valid = Flux.DataLoader((n_datasignals_valid |> Flux.gpu, n_dataprobd_valid |> Flux.gpu), batchsize = batch_size, shuffle = true)

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

    node = NeuralODE(Flux.Chain(Flux.Dense(input_dim, hidden_dim),
                           Flux.Dense(hidden_dim, hidden_dim),
                           Flux.Dense(hidden_dim, hidden_dim),
                           Flux.Dense(hidden_dim, input_dim)) |> Flux.gpu,
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1f-3, abstol = 1f-3, save_start = false) |> Flux.gpu
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)

    return Flux.Chain((x, p=node.p) -> node(x, p),
                 Array,
                 diffeqarray_to_array,
                 Flux.Dense(input_dim, out_dim) |> Flux.gpu), node.p |> Flux.gpu
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
    if iter % 1 == 0
        println("Iteration $iter || Loss = $(loss_node(data.data[1], data.data[2])) || Loss valid = $(loss_node(data_valid.data[1], data_valid.data[2]))")
    end
end

model, parameters = construct_model(2, 2, 10, 5)
opt = Adam(1e-3)

for _ in 1:1
    Flux.train!(loss_node, Flux.params(parameters, model), data, opt, cb = cb)
end

predicted = model(datasignals |> Flux.gpu) |> Flux.cpu
predicted_valid = model(datasignals_valid |> Flux.gpu) |> Flux.cpu

scatter(predicted[1,:], predicted[2,:])
scatter!(predicted_valid[1,:], predicted_valid[2,:])