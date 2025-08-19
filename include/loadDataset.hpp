#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <string>
#include <iostream>
namespace fs = std::filesystem;
class ImageDataset : public torch::data::Dataset<ImageDataset>
{
private:
	std::vector<std::string> class_names_;
	std::vector<std::string> _images;
	std::vector<int> _labels;
public:
	explicit ImageDataset(std::string in_path)
	{
		for (const fs::directory_entry& class_dir : fs::directory_iterator(in_path))
		{
			if (!fs::is_directory(class_dir)) continue;
			size_t label = class_names_.size();
			class_names_.push_back(class_dir.path().filename().string());
			for (const fs::directory_entry& img_path : fs::directory_iterator(class_dir.path()))
			{
				if (img_path.path().extension() != ".png" && img_path.path().extension() != ".jpg") continue;
				_images.push_back(img_path.path().string());
				_labels.push_back(label);
			}
		}
	}
	torch::data::Example<> get(size_t index) override
	{
		cv::Mat mat = cv::imread(_images[index], cv::IMREAD_COLOR);
		cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
		cv::resize(mat, mat, cv::Size(150, 150));
		mat.convertTo(mat, CV_32FC3, 1.0 / 255.0);
		torch::Tensor mat_tensor = torch::from_blob(mat.data, { mat.rows, mat.cols, 3 }, torch::kFloat32).permute({ 2, 0, 1 });
		mat_tensor = torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 })(mat_tensor);
		torch::Tensor label_tensor = torch::full({ 1 }, _labels[index], torch::kInt64);
		return { mat_tensor, label_tensor };
	}
	torch::optional<size_t> size() const override
	{
		return _images.size();
	}
	const std::vector<std::string>& class_names() const
	{
		return class_names_;
	}
};