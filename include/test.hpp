#pragma once
#include "GAN.hpp"
#include "train.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <ctime>
void test(std::string name)
{
    GAN net;
    std::string path = "./models/" + name;
    torch::load(net.gen, path);
    net.gen->to(torch::kCPU);
    net.gen->eval();
    torch::NoGradGuard no_grad;
    torch::Tensor tensor = net.gen->forward(torch::randn({ 150, 100 })).cpu();
    std::string image_name = "./train_save_image/" + std::to_string((int)(time(0))) + ".png";
    saveImage(tensor, image_name);
    std::cout << "照片已保存在了：" << image_name << " 中" << std::endl;
}