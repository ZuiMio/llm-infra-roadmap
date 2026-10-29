# PyTorch Deep Learning & Infrastructure: Q&A Study Notes

---

## 1. PyTorch Debugging & Common Errors

---

## 2. Multi-Layer Perceptrons (MLP) & Training Dynamics

### Q3: Why did my predicted probability distributions (red and blue histograms) completely overlap, and why was accuracy stuck around 65.35%?
* **A:** 
  1. **Vanishing Gradients (Sigmoid in Hidden Layers):** Using `Sigmoid` activation in hidden layers compresses derivatives to a max of `0.25`, causing gradients to vanish in multi-layer networks, effectively paralyzing model learning. Hidden layers should use **ReLU**.
  2. **Lack of Feature Scaling:** The diabetes dataset features have vastly different scales (e.g., pregnancies vs. insulin levels). Without standardization, gradient descent oscillates wildly and fails to converge. 
  3. **Baseline Accuracy Trap:** Since negative classes make up ~65% of the dataset, a model that blindly guesses "0" achieves 65.3% accuracy without learning anything.

### Q4: What do the input and output dimensions (e.g., `8` and `16`) mean in `torch.nn.Linear(8, 16)`, and why change from 6 to 16?
* **A:** 
  * **`8` (Input Dimension):** Represents the number of input features per sample (e.g., 8 features in the diabetes dataset).
  * **`16` (Output Dimension / Neurons):** Represents the number of independent hidden neurons in that layer. Each neuron performs a separate weighted sum and bias calculation to extract a unique pattern from the 8 inputs.
  * Raising it from 6 to 16 provides "dimensionality expansion" (more model capacity and expressive power), though ReLU and normalization are the true drivers of convergence.

### Q5: If we change the hidden layer size back from 16 to 8, how does the model ensure those 8 outputs are correct, and why doesn't it become chaotic?
* **A:** We do not need to manually define what those 8 intermediate outputs represent. Through **backpropagation and gradient descent**, the Loss function acts as a guiding force. It continuously measures the error and precisely adjusts the internal weights so that the intermediate outputs adapt to serve the final classification task. As long as the network capacity is sufficient (e.g., 8 units), the model autonomously discovers useful latent representations.

---

## 3. Training Metrics & Data Loading

### Q6: Why do we reset `running_loss` to 0 every 300 batches instead of accumulating it across the entire epoch?
* **A:** 
  * **Rolling Average vs. Cumulative Average:** If you never reset, the giant loss values from the very beginning of training (when the model is guessing blindly) will permanently taint the average, failing to reflect how well the model is doing *right now*.
  * Resetting every 300 batches creates a **moving window**, giving you a real-time, responsive metric of how the model is performing in the most recent training phase.

### Q7: Is the entire training set treated as a single batch, and what is the relationship between `batch_size`, `DataLoader`, and epochs?
* **A:** No, the training set is not a single batch. 
  * **Batch Size:** The number of samples processed simultaneously in one forward/backward pass (e.g., `batch_size = 64`).
  * **Total Batches per Epoch:** For MNIST (60,000 samples) with `batch_size = 64`, the dataset is split into $60000 / 64 \approx 938$ batches.
  * **DataLoader Mechanism:** It streams data on-demand (`64 × 1 × 28 × 28` per batch) rather than loading everything into memory as one giant 5D tensor.

### Q8: What do the dimensions `64 × 1 × 28 × 28` represent in PyTorch tensor shapes?
* **A:** They follow the standard **`NCHW`** format:
  1. **`64` (Batch Size):** Number of independent image samples processed together.
  2. **`1` (Channels):** Number of color channels (1 for grayscale MNIST, 3 for RGB images).
  3. **`28` (Height):** Vertical pixel count.
  4. **`28` (Width):** Horizontal pixel count.

---

## 4. Convolutional Neural Networks (CNN) Mechanics

### Q9: Does the convolution layer process input channels one by one sequentially, or how does multi-channel convolution work?
* **A:** They are processed **simultaneously and fused**, not sequentially. 
  * A multi-channel kernel has a depth matching the input channels (e.g., 3 slices for RGB). 
  * It computes convolutions across all channels at the exact same time and **sums them up** into a single output value per filter, merging spatial and channel information in one parallel operation.

### Q10: Does `batch_size = 64` mean data is processed sequentially in "64 layers" or queues?
* **A:** No. It means **parallel batch execution**. The GPU treats the 64 images as 64 independent parallel tasks, running them through the exact same convolutional filters simultaneously via optimized matrix multiplication (GEMM), rather than processing them one-by-one in a queue.

### Q11: Is broadcasting happening when a batch of 64 images passes through a convolutional layer?
* **A:** 
  * **Convolution step:** No. This is handled via parallel matrix multiplication (`im2col` + GEMM), where samples are independent.
  * **Bias addition step:** Yes. When adding a 1D bias vector `[C_out]` to a 4D feature map `[B, C_out, H, W]`, PyTorch's **broadcasting mechanism** automatically expands the bias shape to align with the feature map dimensions.

### Q12: Why doesn't normalization (`transforms.Normalize`) use per-batch mean/std instead of global values?
* **A:** 
  * **Unstable distribution:** If every batch used its own dynamic mean and standard deviation, the input scale would constantly shift, making it hard for the model to converge ("building a house on moving sand").
  * **Inference breakdown:** When deployed in production, a user might send a **single image** (Batch Size = 1), where per-batch variance cannot be computed.
  * Therefore, models rely on globally pre-calculated constants (e.g., MNIST mean `0.1307`, std `0.3081`) as a fixed, objective coordinate system.

### Q13: What is the purpose of `x.view(batch_size, -1)` (flattening)?
* **A:** It **flattens** a multi-dimensional spatial tensor (e.g., `[64, 20, 4, 4]`) into a 2D matrix (e.g., `[64, 320]`). 
  * **Why?** Convolutional layers require 4D spatial tensors (`B, C, H, W`), whereas Fully Connected (Linear) layers only accept 2D matrices (`B, Features`) to perform matrix multiplication ($y = Wx + b$).

### Q14: Does flattening scramble or destroy spatial features, and what rule does PyTorch use?
* **A:** **It does not scramble features.** PyTorch uses a strict, deterministic **row-major (C-contiguous)** memory layout:
  * For each sample, it reads channels and spatial pixels in a fixed, repeatable order.
  * Because the flattening rule is 100% deterministic, **the exact same feature always lands at the exact same index** in the 1D vector. 
  * The fully connected layer's weights automatically learn to map these fixed indices to recognize spatial patterns during training.