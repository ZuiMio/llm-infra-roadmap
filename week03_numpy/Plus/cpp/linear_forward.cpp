#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<float> linear_forward_one(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& weights,
    const std::vector<float>& bias
) {
    if (input.empty() || weights.size() != input.size()) {
        throw std::invalid_argument("input length must match weight rows");
    }

    const std::size_t output_features = bias.size();
    if (output_features == 0) {
        throw std::invalid_argument("bias must not be empty");
    }

    for (const auto& row : weights) {
        if (row.size() != output_features) {
            throw std::invalid_argument("all weight rows must match output size");
        }
    }

    std::vector<float> output(output_features, 0.0F);
    for (std::size_t in = 0; in < input.size(); ++in) {
        for (std::size_t out = 0; out < output_features; ++out) {
            output[out] += input[in] * weights[in][out];
        }
    }

    for (std::size_t out = 0; out < output_features; ++out) {
        output[out] += bias[out];
    }
    return output;
}

void scale_in_place(std::vector<float>& values, float scale) {
    for (float& value : values) {
        value *= scale;
    }
}

int main() {
    const std::vector<float> input{1.0F, 2.0F, 3.0F};
    const std::vector<std::vector<float>> weights{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, 1.0F},
    };
    const std::vector<float> bias{0.5F, -0.5F};

    const auto output = linear_forward_one(input, weights, bias);
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i > 0) {
            std::cout << ' ';
        }
        std::cout << output[i];
    }
    std::cout << '\n';

    std::vector<float> values{1.0F, 2.0F};
    scale_in_place(values, 2.0F);
    return 0;
}
