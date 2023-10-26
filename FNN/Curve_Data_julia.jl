using Plots

# Create a grid of points in 2D space
X = LinRange(0.01, 1, 100)  # X-coordinates
Y = LinRange(0.05, 6, 551)  # Y-coordinates
xx, yy = [i for i in X], [j for j in Y]

grid = [i for i in Iterators.product(xx, yy)]

# Extract every tuple in the grid
x = reshape([i[1] for i in grid], 551*100)
y = reshape([i[2] for i in grid], 551*100)

scatter(x, y, markersize=2, legend=false, title="Original Grid", size=(800, 400))

# Apply a nonlinear transformation to create curved data (circular pattern)
theta = atan.(y, x)
r = sqrt.(x .^ 2 .+ y.^ 2)

scatter(r, theta, c=:viridis, title="Curved Data", size=(800, 400), xlabel="X", ylabel="Y")

# Recover the original grid from the curved data
x_recovered = r .* cos.(theta)
y_recovered = r .* sin.(theta)

scatter(x_recovered, y_recovered, markersize=2, legend=false, title="Recovered Grid", size=(800, 400))

# using Plots

# # Create a regular grid of points in 2D space
# n = 100
# x = LinRange(-1, 1, n)  # X-coordinates
# y = LinRange(-1, 1, n)  # Y-coordinates
# xx, yy = [i for i in x], [j for j in y]

# # Define a transformation function from Cartesian to polar coordinates
# function cartesian_to_polar(x, y)
#     r = sqrt(x^2 + y^2)
#     θ = atan(y, x)
#     return r, θ
# end

# # Apply the transformation to map the grid to a circular grid
# circular_x = [cartesian_to_polar(xx[i], yy[j])[1] for i in 1:n, j in 1:n]
# circular_y = [cartesian_to_polar(xx[i], yy[j])[2] for i in 1:n, j in 1:n]

# # Plot the original grid and the circular grid
# plot(
#     heatmap(x, y, xx', c=:viridis, title="Original Grid", size=(800, 400), xlabel="X", ylabel="Y"),
#     heatmap(x, y, circular_x', c=:viridis, title="Circular Grid (R)", size=(800, 400), xlabel="X", ylabel="Y"),
#     heatmap(x, y, circular_y', c=:viridis, title="Circular Grid (Theta)", size=(800, 400), xlabel="X", ylabel="Y"),
# )
using Plots

# Define the number of points in each dimension
n_points = 100

# Define the radius of the circular grid
radius = 2.0

# Create a function to map rectangular coordinates to polar coordinates
function rectangular_to_polar(x, y)
    θ = atan(y, x)
    r = sqrt(x^2 + y^2)  # Normalize to the circular grid
    return r, θ
end

# Create arrays to store the polar coordinates
r_vals = Float64[]
θ_vals = Float64[]

# Create a grid of points and map them to the circular grid
for i in 1:n_points, j in 1:n_points
    x = 2.0 * (i - 0.5) / n_points  # Adjust x coordinate
    y = 2.0 * (j - 0.5) / n_points  # Adjust y coordinate
    r, θ = rectangular_to_polar(x - 1.0, y - 1.0)
    push!(r_vals, r)
    push!(θ_vals, θ)
end

# Plot the circular grid
scatter(θ_vals, r_vals, aspect_ratio=1, legend=false)
plot([0, 2π], [0, radius], grid=false, framestyle=:zerolines)



