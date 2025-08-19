#include "include/loadDataset.hpp"
#include "include/GAN.hpp"
#include "include/test.hpp"
#include "include/train.hpp"
#include <memory>
#include <iostream>
#include <string>
int main()
{
    torch::manual_seed(1);
	torch::set_num_threads(11);
	torch::set_num_interop_threads(11);
    try
    {
        std::ios::sync_with_stdio(false);
        std::cout << "从头开始训练/继续上次的训练/推理（1/2/3）：";
        short is;
        std::cin >> is;
        if (is == 1 || is == 2)
        {
            torch::data::datasets::MapDataset<ImageDataset, torch::data::transforms::Stack<>> datasets = ImageDataset("./train").map(torch::data::transforms::Stack<>());
            std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<ImageDataset, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>> load_datasets = torch::data::make_data_loader(std::move(datasets), torch::data::DataLoaderOptions().batch_size(64).workers(5));
            train(std::move(load_datasets), is);
        }
        else if (is == 3)
        {
            std::string name;
            std::cout << "输入要用来推理的模型，如（G_110_GAN.pt）：";
            std::cin >> name;
            test(name);
        }
        else
        {
            return 1;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    std::cin.get(); std::cin.get();
    return 0;
}