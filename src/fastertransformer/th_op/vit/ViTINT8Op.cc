/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ViTINT8Op.h"

namespace th = torch;
namespace torch_ext {

template class VisionTransformerINT8Func<float>;
template class VisionTransformerINT8Func<half>;

VisionTransformerINT8Class::VisionTransformerINT8Class(std::vector<th::Tensor> w,
                                                       int64_t max_batch,
                                                       int64_t img_size,
                                                       int64_t patch_size,
                                                       int64_t in_chans,
                                                       int64_t embed_dim,
                                                       int64_t num_heads,
                                                       int64_t inter_size,
                                                       int64_t layer_num,
                                                       int64_t int8_mode,
                                                       int64_t with_cls_token):
    st_(w[0].scalar_type()), weights_(w)
{

    // for (int i = 0; i < weights_.size(); i++) {
    //     CHECK_INPUT(weights_[i], st_);
    // }

    output_seq_len_ = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    output_emb_dim_ = embed_dim;

    switch (st_) {
        case at::ScalarType::Float:
            vit_func_ = new VisionTransformerINT8Func<float>(max_batch,
                                                             img_size,
                                                             patch_size,
                                                             in_chans,
                                                             embed_dim,
                                                             num_heads,
                                                             inter_size,
                                                             layer_num,
                                                             1.0f,
                                                             int8_mode,
                                                             with_cls_token,
                                                             weights_);
            break;
        case at::ScalarType::Half:
            vit_func_ = new VisionTransformerINT8Func<half>(max_batch,
                                                            img_size,
                                                            patch_size,
                                                            in_chans,
                                                            embed_dim,
                                                            num_heads,
                                                            inter_size,
                                                            layer_num,
                                                            1.0f,
                                                            int8_mode,
                                                            with_cls_token,
                                                            weights_);

            break;
        default:
            throw std::runtime_error("Wrong th::Tensor type.");
    }
    info_int_ = torch::empty({10}, torch::dtype(torch::kInt64));
    info_int_[0] = max_batch;
    info_int_[1] = img_size;
    info_int_[2] = patch_size;
    info_int_[3] = in_chans;
    info_int_[4] = embed_dim;
    info_int_[5] = num_heads;
    info_int_[6] = inter_size;
    info_int_[7] = layer_num;
    info_int_[8] = int8_mode;
    info_int_[9] = with_cls_token;
}

std::vector<th::Tensor> VisionTransformerINT8Class::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights_);
    tmp.push_back(info_int_);
    return tmp;
}

VisionTransformerINT8Class::~VisionTransformerINT8Class()
{
    delete vit_func_;
}

th::Tensor VisionTransformerINT8Class::forward(th::Tensor input)
{
    CHECK_INPUT(input, st_);
    int batch_size = input.size(0);
    auto output = torch::empty({batch_size, output_seq_len_, output_emb_dim_},
                               torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    vit_func_->forward(batch_size, input, output);
    return output;
}

}  // namespace torch_ext

static auto visionTransformerTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::VisionTransformerINT8Class>("VisionTransformerINT8Class")
#else
    torch::jit::class_<torch_ext::VisionTransformerINT8Class>("VisionTransformerINT8", "Class")
#endif
        .def(torch::jit::init<std::vector<th::Tensor>,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t>())
        .def("forward", &torch_ext::VisionTransformerINT8Class::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::VisionTransformerINT8Class>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::VisionTransformerINT8Class> {
                int state_size = state.size();
                std::vector<th::Tensor>::const_iterator first = state.begin();
                std::vector<th::Tensor>::const_iterator last = state.begin() + (state_size - 1);
                std::vector<th::Tensor> weights(first, last);
                int idx = state.size() - 1;
                int i = 0;
                int64_t max_batch = state[idx][i++].item().to<int>();
                int64_t img_size = state[idx][i++].item().to<int>();
                int64_t patch_size = state[idx][i++].item().to<int>();
                int64_t in_chans = state[idx][i++].item().to<int>();
                int64_t embed_dim = state[idx][i++].item().to<int>();
                int64_t num_heads = state[idx][i++].item().to<int>();
                int64_t inter_size = state[idx][i++].item().to<int>();
                int64_t layer_num = state[idx][i++].item().to<int>();
                int64_t int8_mode = state[idx][i++].item().to<int>();
                int64_t with_cls_token = state[idx][i++].item().to<int>();
                return c10::make_intrusive<torch_ext::VisionTransformerINT8Class>(weights,
                                                                                  max_batch,
                                                                                  img_size,
                                                                                  patch_size,
                                                                                  in_chans,
                                                                                  embed_dim,
                                                                                  num_heads,
                                                                                  inter_size,
                                                                                  layer_num,
                                                                                  int8_mode,
                                                                                  with_cls_token);
            });
