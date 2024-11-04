#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor my_add(
    const torch::Tensor &a, 
    const torch::Tensor &b
);

at::Tensor my_mm(
    const at::Tensor& a, 
    const at::Tensor& b
);

at::Tensor my_mm_bc(
    const at::Tensor& a, 
    const at::Tensor& b,
    const at::Tensor& bias
);

at::Tensor my_func(
    const torch::Tensor &a, 
    const torch::Tensor &b
) {
                
    // CHECK_INPUT(a);
    // CHECK_INPUT(b);

    return my_add(a, b);
}

at::Tensor my_func_mm(
    const at::Tensor& a, 
    const at::Tensor& b
) {

    return my_mm(a, b);
}

at::Tensor my_func_mm_bc(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& bias
    ) {
    
    return my_mm_bc(a, b, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_func", &my_func);
    m.def("my_func_mm", &my_func_mm);
    m.def("my_func_mm_bc", &my_func_mm_bc);
}