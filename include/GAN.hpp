#pragma once
#include <torch/torch.h>
inline void weights_init(torch::nn::Module& module)
{
    if (torch::nn::LinearImpl* linear = module.as<torch::nn::Linear>())
    {
        torch::nn::init::normal_(linear->weight, 0.0, 0.02);
        if (linear->bias.defined()) {
            torch::nn::init::constant_(linear->bias, 0.0);
        }
    }
    else if (torch::nn::BatchNorm1dImpl* batchnorm = module.as<torch::nn::BatchNorm1d>())
    {
        torch::nn::init::normal_(batchnorm->weight, 1.0, 0.02);
        torch::nn::init::constant_(batchnorm->bias, 0.0);
    }
}
class GenImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential model{ nullptr };
public:
    GenImpl()
    {
        model = torch::nn::Sequential(
            torch::nn::Linear(100, 128),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(128, 256),
            torch::nn::BatchNorm1d(256),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(256, 512),
            torch::nn::BatchNorm1d(512),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(512, 1024),
            torch::nn::BatchNorm1d(1024),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(1024, 2048),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(2048, 3 * 150 * 150),
            torch::nn::Tanh()
        );
        model->apply(weights_init);
        register_module("model", model);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor img = model->forward(x);
        img = img.view({ -1, 3, 150, 150 });
        return img;
    }
};
TORCH_MODULE(Gen);
class DisImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential model{ nullptr };
public:
    DisImpl()
    {
        model = torch::nn::Sequential(
            torch::nn::Flatten(),
            torch::nn::Linear(3 * 150 * 150, 1024),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(1024, 512),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(512, 256),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(256, 128),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(128, 64),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
            torch::nn::Linear(64, 1),
            torch::nn::Sigmoid()
        );
        model->apply(weights_init);
        register_module("model", model);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor img_flag = x.view({ x.size(0), -1 });
        torch::Tensor validity = model->forward(img_flag);
        return validity;
    }
};
TORCH_MODULE(Dis);
struct GAN
{
    Gen gen;
    Dis dis;
    GAN() : gen(), dis() {}
};