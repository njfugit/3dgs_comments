#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// --------------------------------------------------------------------------------
// 辅助函数: 单个多通道卷积核的操作 (已更新以支持 stride)
// --------------------------------------------------------------------------------
cv::Mat convolve2d_manual_multi_channel(
    const cv::Mat &image, const cv::Mat &kernel, int padding = 0,
    int stride = 1) // *** CHANGED: 添加 stride 参数 ***
{
  // --- 1. 输入有效性检查 ---
  if (stride < 1) {
    throw std::invalid_argument("Stride 必须大于等于 1");
  }
  int num_channels = image.channels();
  if (num_channels != kernel.channels()) {
    throw std::invalid_argument("输入图像和卷积核的通道数必须相同");
  }

  // --- 2. 应用填充 ---
  cv::Mat padded_image;
  if (padding > 0) {
    cv::copyMakeBorder(image, padded_image, padding, padding, padding, padding,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
  } else {
    padded_image = image.clone();
  }

  // --- 3. 获取尺寸 ---
  int ker_rows = kernel.rows;
  int ker_cols = kernel.cols;

  // --- 4. 计算输出尺寸 (*** CHANGED: 使用 stride 更新公式 ***) ---
  int out_rows = static_cast<int>((padded_image.rows - ker_rows) / stride) + 1;
  int out_cols = static_cast<int>((padded_image.cols - ker_cols) / stride) + 1;

  cv::Mat output = cv::Mat::zeros(out_rows, out_cols, CV_32FC1);

  // --- 5. 核心卷积循环 ---
  for (int y = 0; y < out_rows; ++y) {
    float *output_row = output.ptr<float>(y);
    for (int x = 0; x < out_cols; ++x) {
      float total_sum = 0.0f;
      for (int ky = 0; ky < ker_rows; ++ky) {
        int in_y = y * stride + ky;
        const float *paddimg_row = padded_image.ptr<float>(in_y);
        const float *ker_row = kernel.ptr<float>(ky);
        for (int kx = 0; kx < ker_cols; ++kx) {
          int int_x = x * stride + kx;
          const float *img_p = paddimg_row + int_x * num_channels;
          const float *ker_channel = ker_row + kx * num_channels;
          for (int c = 0; c < num_channels; ++c) {
            total_sum += img_p[c] * ker_channel[c];
          }
        }
      }
      output_row[x] = total_sum;
    }
  }
  return output;
}
// --------------------------------------------------------------------------------
// 上层函数: 模拟一个完整的卷积层 (已更新以支持 stride)
// --------------------------------------------------------------------------------
cv::Mat
convolve_layer_manual(const cv::Mat &image, const std::vector<cv::Mat> &kernels,
                      int padding = 0,
                      int stride = 1) // *** CHANGED: 添加 stride 参数 ***
{
  if (kernels.empty()) {
    throw std::invalid_argument("卷积核组不能为空");
  }

  std::vector<cv::Mat> output_channels;
  output_channels.reserve(kernels.size());

  for (const auto &kernel : kernels) {
    // *** CHANGED: 将 stride 参数传递给辅助函数 ***
    cv::Mat single_feature_map =
        convolve2d_manual_multi_channel(image, kernel, padding, stride);
    output_channels.push_back(single_feature_map);
  }

  cv::Mat final_output;
  cv::merge(output_channels, final_output);

  return final_output;
}

int main() {
  // --- 创建一个3通道的输入图像 (5x5x3) ---
  cv::Mat image(5, 5, CV_32FC3, cv::Scalar(1, 10, 100));
  // 为了让结果更直观，我们用一个更变化的图像
  cv::Mat b_channel =
      (cv::Mat_<float>(5, 5) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25);
  cv::Mat g_channel = b_channel * 2;
  cv::Mat r_channel = b_channel * 3;
  std::vector<cv::Mat> channels = {b_channel, g_channel, r_channel};
  cv::merge(channels, image);

  // --- 创建一个3通道的卷积核 ---
  cv::Mat kernel_1(3, 3, CV_32FC3, cv::Scalar(0, 0, 0));
  std::vector<cv::Mat> k1_channels = {
      cv::Mat::ones(3, 3, CV_32F),     // B通道权重
      cv::Mat::zeros(3, 3, CV_32F),    // G通道权重
      cv::Mat::ones(3, 3, CV_32F) * -1 // R通道权重
  };
  cv::merge(k1_channels, kernel_1);
  std::vector<cv::Mat> kernels = {kernel_1}; // 我们只用一个核来演示

  std::cout << "--- 演示 Stride > 1 的效果 ---" << std::endl;
  std::cout << "输入图像尺寸: " << image.size() << std::endl;

  // --- 示例 1: Stride = 1 (结果与之前相同) ---
  cv::Mat layer_output_s1 = convolve_layer_manual(image, kernels, 1, 1);
  std::cout << "\n--- Stride = 1, Padding = 1 ---" << std::endl;
  std::cout << "输出尺寸: " << layer_output_s1.size() << std::endl;
  std::cout << "输出结果:\n" << layer_output_s1 << std::endl;

  // --- 示例 2: Stride = 2 ---
  cv::Mat layer_output_s2 = convolve_layer_manual(image, kernels, 1, 2);
  std::cout << "\n--- Stride = 2, Padding = 1 ---" << std::endl;
  std::cout << "输出尺寸: " << layer_output_s2.size() << std::endl;
  std::cout << "输出结果:\n" << layer_output_s2 << std::endl;

  // 解释输出尺寸:
  // 输入5x5, padding=1 -> 填充后7x7
  // 核3x3
  // 输出尺寸 = (7 - 3) / 2 + 1 = 4 / 2 + 1 = 3
  // 所以输出尺寸应该是 3x3, 与程序结果一致。

  return 0;
}