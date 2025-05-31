# Mini-CNN with XNNPACK

* 3 × 3 Conv (2 channels) + ReLU  
* 28 × 28 Average-Pool → 1 × 1 feature map  
* Fully-connected layer to 3 logits  
* Soft-max for probabilities

Everything is pure **C++17 + XNNPACK**—no other deps.

## Build & run
```bash
git clone https://github.com/your-user/xnnpack-mini-cnn.git
cd xnnpack-mini-cnn
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./mini_cnn