### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f2822854-c427-4160-b492-fc02e02597b7
using PlutoUI

# ╔═╡ ac970d17-728f-4a66-aac5-a3eb2bfc1cbf
md"Después de realizar la reducción de dimensionalidad para los datos de señales y las distribuciones de probabilidad mediante PCA, procederemos a realizar un mapeo entre ambas utilizando una red neuronal de arquitectura feedforward.

El análisis de componentes principales para los datos de 55100 señales asociadas a una longitud de correlación media dada, y a un error dado se obutvo que el 98% de la varianza total se encuentra en solo 2 direcciones principales y el 100% en 3 componentes principales. 

Teniendo esto en cuenta se diseñó una red neuronal con arquitectura feedforward. Esta red consta de una capa de entrada con 3 neuronas para las 3 componentes principales de las señales, seguida de 3 capas ocultas con 25, 32 y 25 neuronas respectivamente y una capa de salida con 2 neuronas para los parámetros asociados a la distribucion de probabilidad de donde se obtienen las señales.

Para la capa de entrada, se utilizaron los datos de las señales transformados en las 3 componentes principales, mientras que en la capa de salida, se utilizaron los datos de los parámetros $\sigma$ y $l_{cm}$ de las distribuciones de probabilidad simuladas para el calculo de las señales."

# ╔═╡ 61fb2868-01d7-4785-9e8d-c795e6ba517c
# using Flux
# using Statistics
# using Flux: train!
# using Plots
# using Distributions
# using ProgressMeter
# using MultivariateStats
# using DataFrames
# using CSV
# using StatsPlots
# using LaTeXStrings
# using LinearAlgebra
# using PlotlyJS
# using CUDA
# using Random

# ╔═╡ dc468a74-a0ee-4797-b37c-5cab75de79dd
begin
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
	
end

# ╔═╡ cb3c4860-2f12-4498-90bb-0bec74d4db21
# Distribucion de probabilidad log-normal
function Pln(lcm::Float32, σ::Float32)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

# ╔═╡ 5c50de00-61b7-47c8-bb3d-884979a0baca
# Leemos los datos a los que les realizamos PCA de las señales

#path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"

#df_datasignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)


# ╔═╡ ea439340-2d5a-42f0-8fbe-e02567f342ea
md"Funciones de pre procesamiento para el escaleo de los datos"

# ╔═╡ d6625e71-6832-4343-a1e5-2ce09e3f54ee
begin
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
end

# ╔═╡ 387d7560-0fbb-4a1f-abec-ca6961a2ba6b
md"Metricas de validacion de la red neuronal"

# ╔═╡ e9ea710d-4712-457f-add7-87ceb06d8c42
begin
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
	
end

# ╔═╡ 8d191f45-c2d8-44da-a748-261810ff0691
md"Regularizaciones L1 y L2 para la red neuronal"

# ╔═╡ bcb38307-21d4-443f-8cb5-fb98fc905d4d
begin
	
	pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
	pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)
end

# ╔═╡ 174c54ec-9022-4acb-b49f-6676274fac0a
md"Primero, utilizamos la técnica de validación cruzada k-folds para asegurarnos de que no haya un sobreajuste en la red neuronal. Dividimos los datos en 5 conjuntos (folds) disjuntos, cada uno representando el 10% del conjunto de datos total. Luego, entrenamos 5 modelos idénticos, evaluamos las predicciones de los datos de validación utilizando métricas como MAE, RMSE y R2. También almacenamos los datos de validación y sus predicciones. Después de las 5 repeticiones, graficamos las evaluaciones de las predicciones de los datos de validación para cada fold y observamos que los valores son similares a los obtenidos en entrenamientos anteriores.

Por lo tanto hay una estabilidad en las predicciones ya que los resultados son consistentes entre las repeticiones y folds sugiriendo que el modelo de la red es estable y no está sobreajustando. La consistencia en las metricas de la evaluación tambien indica que se mantiene una rendimiento similar en diferentes subconjuntos de validación. Ahora bien necesitariamos metricas de evaluación adecuadas para poder asegurar que este modelo de red nueronal es adecuado para esta tarea de predicción.
"

# ╔═╡ 967c08af-6701-49b9-8c95-1a16f9bc5cfa
PlutoUI.Resource("https://imgur.com/E8se1w3.png")

# ╔═╡ aee65a9d-1e12-4342-abf8-e23ed7109f03
md"Vamos a entrenar por ultimo el modelo utilizando el 10% de datos como validación para una última evaluación. Sin embargo, también es una buena idea considerar la posibilidad de entrenar el modelo con todos los datos disponibles para aprovechar al máximo el conjunto de entrenamiento y testear en datos nuevos.

Además agregamos una función de costo compuesta que compara las distribuciones de probabilidad calculadas con los parámetros predichos frente a los parámetros reales. Esta función de pérdida adicional permite cuantificar la similitud entre las distribuciones de probabilidad predichas y las reales, que es lo especialmente relevante en aplicaciones donde la precisión de las distribuciones de probabilidad es crucial.

Al evaluar el modelo con esta función de pérdida, puedes obtener información adicional sobre la calidad de las predicciones en términos de la forma y la precisión de las distribuciones de probabilidad."

# ╔═╡ a3737772-fbaf-4027-bae9-a98058033caa
md"Presentamos los resultados de la función costo en función de las épocas de entrenamiento donde se observa como tanto el loss en los datos de entrenamiento y validación disminuyen en paralelo."

# ╔═╡ 3d85adb0-66d2-4b03-a999-8ac01b3f083b
PlutoUI.Resource("https://imgur.com/D8DCtse.png")

# ╔═╡ ee6a6c8e-f814-430a-b009-4acac8022a97
md"Presentamos ademas el error absoluto medio en las predicciones de los parámetros, notando zonas en las que las predicciones son mucho peores."

# ╔═╡ df0af5bd-4746-4601-8be8-a9159307e0f9
PlutoUI.Resource("https://imgur.com/KGzTnns.png")

# ╔═╡ 9cee333d-ff77-4a3e-84f7-29f265bf9a77
md"Esto mismo se puede hacer para el grafico de los datos de entrada donde solo se grafican dos componentes principales."

# ╔═╡ df5701bc-4aea-47c1-8a84-77dc576e59d5
PlutoUI.Resource("https://imgur.com/byi0vIn.png")

# ╔═╡ 16a74685-d249-4d37-9c5a-c96f58d5cd93
md"Para notar mas errores en otras zonas se quitan los datos donde hay mas error absoluto que pueden tener que ver con la resolución utilizada tanto en el intervalo de tiempo total de las señales como en el número de compartimientos utilizado para las distribuciones de probabilidad."

# ╔═╡ 3479d072-c92a-4ec4-9439-c4f6bef6a70c
PlutoUI.Resource("https://imgur.com/ZbuLjTq.png")

# ╔═╡ d71d5c8c-9f2b-4c47-8af3-f17097568e21
PlutoUI.Resource("https://imgur.com/yxyUuPV.png")

# ╔═╡ e669761d-4011-46e9-93b4-b0e0636e28ea
md"# Misma red con activación tanh" 

# ╔═╡ 16c02915-6209-47e0-95ac-bed6cdb8629d
md"El mismo procedimiento mendionado anteriormente pare esta otra red, en este caso solo usando la métrica Error absoluto medio relativo RMAE punto a punto como

$\frac{\frac{1}{2} (|σ_{true} - σ_{pred}| + |lcm_{true} - lcm_{pred}|)}{ \frac{1}{2} (\sigma_{true} + lcm_{true})}$

que capta las zonas donde la red falla en la zona de valores de σ y lcm pequeños.
"

# ╔═╡ 6fc7dcd3-967c-4e4a-9a3f-bd20b7e2bf38
PlutoUI.Resource("https://imgur.com/8BeLav3.png")

# ╔═╡ 166e9610-b7ab-4043-b211-7ca60e8fab88
PlutoUI.Resource("https://imgur.com/03AhwOU.png")

# ╔═╡ d39bb182-b8ac-4b8d-9eb8-4fecc7cc941f
PlutoUI.Resource("https://imgur.com/ROn4ViY.png")

# ╔═╡ bfdcc9a1-3f66-4003-a4d5-b4aeb7e219a9
PlutoUI.Resource("https://imgur.com/TbdAOp0.png")

# ╔═╡ 45f0c928-79a8-46b8-873d-b6ab43625ec7
md"La función de activación tanh redujo la función costo pero no lo suficiente como para tener mejorias notorias en las predicciones, sigue habiendo zonas donde la red no puede predecir correctamente los parámetros."

# ╔═╡ 2a1f773b-abda-47b4-9c79-90fe8e45a1ec
md"# Red 2 una red mucho mas compleja"

# ╔═╡ 6db9d0e7-084d-4082-8d6a-7db3a4a3a170
PlutoUI.Resource("https://imgur.com/jNAWg2G.png")

# ╔═╡ e183ddd4-a0d2-4181-9680-f61f3d4d1440
PlutoUI.Resource("https://imgur.com/aVeX19L.png")

# ╔═╡ bfa0c450-b1d0-435c-8747-a3a222c18937
PlutoUI.Resource("https://imgur.com/bVNHwhQ.png")

# ╔═╡ c7a011e8-9e2c-47a1-8a3f-6f9d3cc13e62
PlutoUI.Resource("https://imgur.com/y8OJHyp.png")

# ╔═╡ 3fe87256-a6f8-4817-90da-f3b423d318c6
md"Por lo que muestran los errores esta red funciona mucho peor que la anteiror a pesar de ser mas compleja debido a la cantidad excesiva de paráetros con respecto a datos que se tienen."

# ╔═╡ 559aabbf-7a57-47ab-aa55-82ea4617a2c0
md"# Red 3 una red un poco mas compleja
ya que el extremo de una red mucho mas copleja no mejora los resultados probamos con una red un poco mas compleja añadiendo una capa mas y aumentando el número de neuronas
"

# ╔═╡ 77807a80-2ae4-4d25-a412-9fdbc6ea7bd7
PlutoUI.Resource("https://imgur.com/vooLaGL.png")

# ╔═╡ 029aaf3e-b5a5-4499-8068-a50be03073cc
PlutoUI.Resource("https://imgur.com/FmpUNwV.png")

# ╔═╡ 9fb64056-c9bc-472e-80b1-36cc398c4622
PlutoUI.Resource("https://imgur.com/aBs5thx.png")

# ╔═╡ 655a2121-515e-4537-b74d-fc958a7ef645
PlutoUI.Resource("https://imgur.com/3hlYRx9.png")

# ╔═╡ 633377da-a201-4177-95c6-0bde4cef1616
md"Esta red es un poco mas compleja que la anterior y logra disminuir tanto el minimo al que llega la función costo de entrenamiento como la de validación, además mejora los resultados obtenidos pero se repite sistematicamente el hecho de tener zonas donde la red tiene un error mas grande, quizas una arquitectura óptima se encuentre en una red similar a esta. Evaluamos ademas esta red en una zona donde predice los parámetros correctamente tomando todos los casos en los que $\sigma > 0.5$"

# ╔═╡ b45b0fb1-a372-4f15-a5a6-682695a51490
md"# En zona $\sigma > 0.5$" 

# ╔═╡ c1f6594c-d88e-46f1-ae82-3fa81b8a908d
PlutoUI.Resource("https://imgur.com/bQFS3pC.png")

# ╔═╡ 7678aecc-7f95-4c53-9b52-cd5765a32265
PlutoUI.Resource("https://imgur.com/t1F8dlT.png")

# ╔═╡ 2f243e4b-981f-486e-8d27-92562562b335
PlutoUI.Resource("https://imgur.com/Zcm4gXo.png")

# ╔═╡ 2edc9a7f-5719-4cec-9f59-43bea0ace516
PlutoUI.Resource("https://imgur.com/Yv3UuRz.png")

# ╔═╡ 7c39b029-e65a-4120-b6f3-8f93b3096f32
PlutoUI.Resource("https://imgur.com/BV0IxpD.png")

# ╔═╡ 71cbdcb7-7ba1-4324-a1e7-d25646130d84
PlutoUI.Resource("https://imgur.com/WCMTO4T.png")

# ╔═╡ dcd6fcf5-9b00-4a54-8576-66e7aa740051
md"Se puede observar que en esta zona la red predice los parámetros con un error máximo del 3% para ciertos valores y por lo observado en los gráficos ocurre para valores de lcm cercanos a 0.5 y para valores de $\sigma$ cercanos a 0.5"

# ╔═╡ 468c88ef-ba47-46cb-ac6d-6bcbfe0e4617
model = Chain(
	Dense(3, 25, tanh),
	Dense(25, 32, tanh),
	Dense(32, 25, tanh),
	Dense(25, 2, softplus),
)

# ╔═╡ 17d3a11e-742e-4f60-9c77-bb08e4f8ee43
# ╠═╡ skip_as_script = true
#=╠═╡
begin
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
	        Dense(3, 25, relu),
	        Dense(25, 32, relu),
	        Dense(32, 25, relu),
	        Dense(25, 2, softplus),
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
	
	    # Entrenamos la red neuronal con el loss mse dividiendo en 10 la tasa de aprendizaje cada 1000 epocas
	    for epoch in 1:epochs
	        Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
	        if epoch % 1000 == 0
				η /= 10
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
end
  ╠═╡ =#

# ╔═╡ 5ec4b590-7fe8-409e-955e-c5619d8269e5
#=╠═╡
begin
	num_datos = Int(size(df_datasignals, 1))
	step_valid = 20
	
	datasignals_valid = Float32.(Matrix(df_datasignals[1:step_valid:num_datos,1:3])')
	datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),1:3])')
	
	σ_valid = df_datasignals[1:step_valid:num_datos,4]
	lcm_valid = df_datasignals[1:step_valid:num_datos,5]
	
	σ_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),4]
	lcm_col = df_datasignals[setdiff(1:num_datos, 1:step_valid:num_datos),5]
	
	dataparams = hcat(lcm_col, σ_col)'
	dataparams_valid = hcat(lcm_valid, σ_valid)'
	
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
	
	# Loss compuesto
	function composed_loss(x,y)
	    y_hat = model(x)
	    Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
	    Pln_real = Pln.(y[1,:], y[2,:])
	    return mean(Flux.mse.(Pln_predicted,Pln_real)) + Flux.mse(y_hat, y)
	end
	
	# Definimos el modo de aprendizaje y la tasa de aprendizaje
	η = 1e-4
	opt = ADAM(η)
	
	# Definimos el batch size
	batch_size = 1101
	
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
	epochs = 10000
	
	# Entrenamos la red neuronal con el loss mse
	for epoch in 1:epochs
	    Flux.train!(loss, Flux.params(model, opt), data, opt, cb=cb)
	    if epoch % 1000 == 0
	        η /= 10
	        opt = ADAM(η)
	    end
	end
	
	epochs = 50
	
	# Entrenamos la red neuronal con el loss compuesto para mejorar la predicción de los parámetros
	for epoch in 1:epochs
	    Flux.train!(composed_loss, Flux.params(model, opt), data, opt, cb=cb2)
	end
	
	# Plots de la función de loss
	pl_loss = Plots.plot(losses[200:end], xlabel = "Épocas", ylabel = "Loss", label = "Datos de entrenamiento",tickfontsize=12, labelfontsize=14, legendfontsize=12, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, size = (600,400))
	Plots.plot!(losses_valid[200:end], xlabel = "Épocas", ylabel = "Loss", label = "Datos de validación")
end
  ╠═╡ =#

# ╔═╡ 6abfbcaf-397a-461f-8117-912b4a23e7c7
model = Chain(
	Dense(3, 30, tanh),
	Dense(30, 25, tanh),
	Dense(25, 32, tanh),
	Dense(32, 25, tanh),
	Dense(25, 2, softplus),
)

# ╔═╡ ebe1da4b-d6f4-43e9-a49a-45b12b5c3e90
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "f5c06f335ceddc089c816627725c7f55bb23b077"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═f2822854-c427-4160-b492-fc02e02597b7
# ╟─ac970d17-728f-4a66-aac5-a3eb2bfc1cbf
# ╠═61fb2868-01d7-4785-9e8d-c795e6ba517c
# ╠═dc468a74-a0ee-4797-b37c-5cab75de79dd
# ╠═cb3c4860-2f12-4498-90bb-0bec74d4db21
# ╠═5c50de00-61b7-47c8-bb3d-884979a0baca
# ╟─ea439340-2d5a-42f0-8fbe-e02567f342ea
# ╠═d6625e71-6832-4343-a1e5-2ce09e3f54ee
# ╟─387d7560-0fbb-4a1f-abec-ca6961a2ba6b
# ╠═e9ea710d-4712-457f-add7-87ceb06d8c42
# ╟─8d191f45-c2d8-44da-a748-261810ff0691
# ╠═bcb38307-21d4-443f-8cb5-fb98fc905d4d
# ╟─174c54ec-9022-4acb-b49f-6676274fac0a
# ╠═17d3a11e-742e-4f60-9c77-bb08e4f8ee43
# ╠═967c08af-6701-49b9-8c95-1a16f9bc5cfa
# ╟─aee65a9d-1e12-4342-abf8-e23ed7109f03
# ╠═5ec4b590-7fe8-409e-955e-c5619d8269e5
# ╟─a3737772-fbaf-4027-bae9-a98058033caa
# ╠═3d85adb0-66d2-4b03-a999-8ac01b3f083b
# ╟─ee6a6c8e-f814-430a-b009-4acac8022a97
# ╠═df0af5bd-4746-4601-8be8-a9159307e0f9
# ╟─9cee333d-ff77-4a3e-84f7-29f265bf9a77
# ╠═df5701bc-4aea-47c1-8a84-77dc576e59d5
# ╟─16a74685-d249-4d37-9c5a-c96f58d5cd93
# ╠═3479d072-c92a-4ec4-9439-c4f6bef6a70c
# ╠═d71d5c8c-9f2b-4c47-8af3-f17097568e21
# ╟─e669761d-4011-46e9-93b4-b0e0636e28ea
# ╟─16c02915-6209-47e0-95ac-bed6cdb8629d
# ╠═468c88ef-ba47-46cb-ac6d-6bcbfe0e4617
# ╠═6fc7dcd3-967c-4e4a-9a3f-bd20b7e2bf38
# ╠═166e9610-b7ab-4043-b211-7ca60e8fab88
# ╠═d39bb182-b8ac-4b8d-9eb8-4fecc7cc941f
# ╠═bfdcc9a1-3f66-4003-a4d5-b4aeb7e219a9
# ╟─45f0c928-79a8-46b8-873d-b6ab43625ec7
# ╟─2a1f773b-abda-47b4-9c79-90fe8e45a1ec
# ╠═ebe1da4b-d6f4-43e9-a49a-45b12b5c3e90
# ╠═6db9d0e7-084d-4082-8d6a-7db3a4a3a170
# ╠═e183ddd4-a0d2-4181-9680-f61f3d4d1440
# ╠═bfa0c450-b1d0-435c-8747-a3a222c18937
# ╠═c7a011e8-9e2c-47a1-8a3f-6f9d3cc13e62
# ╟─3fe87256-a6f8-4817-90da-f3b423d318c6
# ╟─559aabbf-7a57-47ab-aa55-82ea4617a2c0
# ╠═6abfbcaf-397a-461f-8117-912b4a23e7c7
# ╠═77807a80-2ae4-4d25-a412-9fdbc6ea7bd7
# ╠═029aaf3e-b5a5-4499-8068-a50be03073cc
# ╠═9fb64056-c9bc-472e-80b1-36cc398c4622
# ╠═655a2121-515e-4537-b74d-fc958a7ef645
# ╟─633377da-a201-4177-95c6-0bde4cef1616
# ╟─b45b0fb1-a372-4f15-a5a6-682695a51490
# ╠═c1f6594c-d88e-46f1-ae82-3fa81b8a908d
# ╠═7678aecc-7f95-4c53-9b52-cd5765a32265
# ╠═2f243e4b-981f-486e-8d27-92562562b335
# ╠═2edc9a7f-5719-4cec-9f59-43bea0ace516
# ╠═7c39b029-e65a-4120-b6f3-8f93b3096f32
# ╠═71cbdcb7-7ba1-4324-a1e7-d25646130d84
# ╟─dcd6fcf5-9b00-4a54-8576-66e7aa740051
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
