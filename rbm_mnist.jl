import JLD2
using MLDatasets: MNIST
using RestrictedBoltzmannMachines: RBM, Binary, xReLU, initialize!, log_pseudolikelihood, pcd!
using StandardizedRestrictedBoltzmannMachines: standardize
using CudaRBMs: gpu, cpu
using Statistics: mean
using Optimisers: Adam

train_data = MNIST(:train).features .> 0.5
rbm = RBM(Binary((28, 28)), xReLU((100,)), zeros(28, 28, 100))
initialize!(rbm, train_data)
rbm = gpu(standardize(rbm))

function callback(; rbm, iter, _...)
    if iszero(iter % 500)
        Δt = @elapsed (lpl = mean(log_pseudolikelihood(cpu(rbm), cpu(train_data))))
        @info iter lpl Δt
    end
end

state, ps = pcd!(
    rbm, gpu(train_data);
    optim = Adam(1f-5, (0f0, 999f-3), 1f-6),
    steps = 50,
    batchsize = 512,
    iters = 50000,
    l2l1_weights = 0.001,
    ϵv = 1f-1, ϵh = 0f0, damping = 1f-1,
    callback
)

rbm = cpu(rbm)
JLD2.jldsave("rbm.jld2"; rbm)
