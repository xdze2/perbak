# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Julia 1.5.1
#     language: julia
#     name: julia-1.5
# ---

using Dates
using JLD

using CUDA
CUDA.name(CuDevice(0))

# https://github.com/JuliaGPU/CUDA.jl/blob/master/examples/pairwise.jl

function topple(a, b, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        @inbounds a_new = b[i, j] # copy
        if a_new > 5
            a_new -= 4
        end
        @inbounds i > 1  && (b[i-1, j] > 5) && (a_new += 1)
        @inbounds j > 1  && (b[i, j-1] > 5) && (a_new += 1)
        @inbounds i < n  && (b[i+1, j] > 5) && (a_new += 1)
        @inbounds j < n  && (b[i, j+1] > 5) && (a_new += 1)
        
        @inbounds a[i, j] = a_new
    end

    return
end

function topple25k(a, b, n_thread, n_block)
    n = n_thread*n_block
    for k=1:25000
        CUDA.@sync begin
            @cuda threads=(n_thread, n_thread) blocks=(n_block, n_block) topple(a, b, n)
        end
        CUDA.@sync begin
            @cuda threads=(n_thread, n_thread) blocks=(n_block, n_block) topple(b, a, n)
        end
    end
end

# +
# init
n_block = 50
n_thread = 20
n = n_block*n_thread
println(n)

z = rand(10:120, n, n);
a = CuArray(z);
b = CuArray(copy(z));
println("sum: ", sum(a))
println("target sum: ", Int(round((2*0.084+3*0.188+4*0.312+5*0.416)*n^2)))
# -

for k in 1:500
    topple25k(a, b, n_thread, n_block)
    println(sum(a), " ", now())
    if a==b
        break
    end
end
print("done")

path = "./stable/stable_$(n)x$(n)_$(now()).jld"
save(path, "a", Array(a))
println(path, " saved!")

# +
## Risk map
# -

function topple_count(a, b, c, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        @inbounds a_new = b[i, j] # copy
        if a_new > 5
            a_new -= 4
            c[i, j] += 1
        end
        @inbounds i > 1  && (b[i-1, j] > 5) && (a_new += 1)
        @inbounds j > 1  && (b[i, j-1] > 5) && (a_new += 1)
        @inbounds i < n  && (b[i+1, j] > 5) && (a_new += 1)
        @inbounds j < n  && (b[i, j+1] > 5) && (a_new += 1)
        
        @inbounds a[i, j] = a_new
    end

    return
end

function topple_count_N(a, b, c, n_thread, n_block, N)
    n = n_thread*n_block
    for k=1:N
        CUDA.@sync begin
            @cuda threads=(n_thread, n_thread) blocks=(n_block, n_block) topple_count(a, b, c, n)
        end
        CUDA.@sync begin
            @cuda threads=(n_thread, n_thread) blocks=(n_block, n_block) topple_count(b, a, c, n)
        end
    end
end

# +
#data = load("./stable800.jld")
data = load(path)

a = Array(data["a"]);
print(size(a))
# -

typeof(a)

c = CuArray(zeros(Float32, size(a)...));
a = CuArray(a)
z = CuArray(copy(a));
b = CuArray(copy(z));

for i in CartesianIndices(z)
    copyto!(b, z);
    b[i] += 1
    N = 2
    for k in 1:5000
        topple_count_N(a, b, c, n_thread, n_block, N)
        if a == b
            break
        end
        N *= 2
    end
    print("$(i)", " $N ", now(), "\r")
end

path = "./stable/count_$(n)x$(n)_$(now()).jld"
save(path, "c", c)
println(path, " saved!")

using Luxor
# http://juliagraphics.github.io/Luxor.jl/stable/examples/
using ColorSchemes

# +
# ==========
#  Graphics
# ==========
# Colors map functions :
function z_scale(zi)
    # colors = reverse(ColorDict["acw"]["Phaedra"])
    # https://randyzwitch.com/NoveltyColors.jl/acw.html
    colors = ["#FFFAD5", "#FFFF9D", "#BEEB9F", "#79BD8F", "#00A388", "#FF6138", "#D90000"] # EC2E49
    color_idx = min(zi + 1, length(colors))
    return colors[color_idx]
end


function scalar_scale(s)
    return get(ColorSchemes.thermal, s)
end


function draw(z; px_size=12, margin=1,
                 color_scale=z_scale,
                 savepath="temp/temporary.png")
    @png begin
        n, m = size(z)
        w = m*(px_size + margin)
        h = n*(px_size + margin)
        Drawing(w, h, savepath)
        background("white")
        for i in CartesianIndices(z)
            setcolor(color_scale(z[i]))
            corner = (i.I .- 1).*(px_size + margin) 
            rect(corner[2], corner[1], px_size, px_size, :fill)
        end
    end
end
# -

draw(c./maximum(c); px_size=1, margin=0, color_scale=scalar_scale,
     savepath="./stable/count_$(n)x$(n)_$(now()).png")






