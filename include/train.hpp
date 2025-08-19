#pragma once
#include "GAN.hpp"
#include "loadDataset.hpp"
#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <cstdlib>
namespace fs = std::filesystem;
int findEpoch(std::string path)
{
    int i;
    for (i = 0; i < path.size(); ++i)
    {
        if (path[i] == '_') break;
    }
    std::string num_str = "";
    for (int j = i + 1; j < path.size(); ++j)
    {
        if (path[j] == '_') break;
        else
        {
            num_str += path[j];
        }
    }
    return atoi(num_str.c_str());
}
void eraseModel(std::string file_name)
{
    for (const fs::directory_entry& model : fs::directory_iterator("./models"))
    {
        if (model.path().filename().string() == file_name)
        {
            fs::remove(model.path());
        }
    }
}
void saveImage(torch::Tensor tensor, const std::string& path)
{
    tensor = tensor.permute({ 0, 2, 3, 1 });
    tensor = tensor.mul(0.5).add(0.5).mul(255).to(torch::kUInt8);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img);
}
void train(std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<ImageDataset, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>> data_load, const short& is)
{
    GAN net;
    torch::Device de = torch::kCPU;
    if (torch::cuda::is_available()) de = torch::kCUDA;
    if (de == torch::kCUDA) std::cout << "CUDA" << std::endl;
    else std::cout << "CPU" << std::endl;
    int epoch;
    if (is == 1)
    {
        net.gen->to(de);
        net.dis->to(de);
        net.gen->train();
        net.dis->train();
        epoch = 0;
    }
    else
    {
        std::string path1, path2;
        std::ifstream in("./models/theNewestModel.txt");
        std::getline(in, path1);
        std::getline(in, path2);
        torch::load(net.gen, path1);
        torch::load(net.dis, path2);
        in.close();
        net.gen->to(de);
        net.dis->to(de);
        net.gen->train();
        net.dis->train();
        epoch = findEpoch(path1);
    }
    torch::optim::Adam g_op(net.gen->parameters(), torch::optim::AdamOptions(0.000007).betas({ 0.5, 0.999 }).weight_decay(1e-5));
    torch::optim::Adam d_op(net.dis->parameters(), torch::optim::AdamOptions(0.000007).betas({ 0.5, 0.999 }).weight_decay(1e-5));
    std::cout << "开始训练" << std::endl;
    std::cout << "如果要暂停训练关掉此窗口即可" << std::endl;
    std::ofstream out("./models/model.log", std::ios::trunc);
    while (true)
    {
        ++epoch;
        double d_running_loss = 0.0;
        double g_running_loss = 0.0;
        int count = 0;
        for (torch::data::Example<>& batch : *data_load)
        {
            torch::Tensor real_image = batch.data.to(de);
            net.dis->zero_grad();
            torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
            torch::Tensor real_output = net.dis->forward(real_image);
            torch::Tensor real_d_loss = torch::binary_cross_entropy(real_output, real_labels);
            real_d_loss.backward();
            torch::Tensor noise = torch::randn({ batch.data.size(0), 100 });
            torch::Tensor fake_image = net.gen->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
            torch::Tensor fake_output = net.dis->forward(fake_image.detach());
            torch::Tensor fake_d_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            fake_d_loss.backward();
            torch::Tensor d_loss = real_d_loss + fake_d_loss;
            d_running_loss += d_loss.item<float>();
            d_op.step();
            net.gen->zero_grad();
            fake_labels.fill_(1);
            fake_output = net.dis->forward(fake_image);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            g_running_loss += g_loss.item<float>();
            g_op.step();
            ++count;
        }
        std::cout << "Epoch: " << epoch << '\t' << "G_loss: " << g_running_loss / (double)count << '\t' << "D_loss: " << d_running_loss / (double)count << std::endl;
        if (epoch % 10 == 0)
        {
            std::string G_name = "G_" + std::to_string(epoch) + "_GAN.pt";
            std::string path1 = "./models/" + G_name;
            torch::save(net.gen, path1);
            out << G_name << '\t' << "G_loss: " << g_running_loss << std::endl;
            std::string D_name = "D_" + std::to_string(epoch) + "_GAN.pt";
            std::string path2 = "./models/" + D_name;
            torch::save(net.dis, path2);
            std::string erase_model_gen = "G_" + std::to_string(epoch - 50) + "_GAN.pt";
            std::string erase_model_dis = "D_" + std::to_string(epoch - 50) + "_GAN.pt";
            eraseModel(erase_model_gen);
            eraseModel(erase_model_dis);
            std::ofstream out2("./models/theNewestModel.txt", std::ios::trunc);
            out << D_name << '\t' << "D_loss: " << d_running_loss << std::endl;
            out2 << path1 << std::endl << path2 << std::endl;
            out2.close();
            std::string image_name = "G_" + std::to_string(epoch) + ".png";
            saveImage(net.gen(torch::randn({ 150, 100 })), "./train_save_image/" + image_name);
            std::cout << G_name << "已保存，" << D_name << "已保存" << std::endl;
        }

    }
    out.close();
}