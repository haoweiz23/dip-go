import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
import pickle
import math
from flops import count_ops_and_params
from tqdm import tqdm
from typing import Optional, List

THE_END_OF_BRANCH = [
    ("down", "attentions", 0, 0, 1),
    ("down", "attentions", 0, 1, 2),
    ("down", "downsampler", 0, 2, 3),
    ("down", "attentions", 1, 0, 4),
    ("down", "attentions", 1, 1, 5),
    ("down", "downsampler", 1, 2, 6),
    ("down", "attentions", 2, 0, 7),
    ("down", "attentions", 2, 1, 8),
    ("down", "downsampler", 2, 2, 9),
    ("down", "resnet", 3, 0, 10),
    ("down", "resnet", 3, 1, 11),
    ("mid", "mid_block", 0, 0, 12),
    ("up", "resnet", 3, 2, 11),
    ("up", "resnet", 3, 1, 10),
    ("up", "upsampler", 3, 0, 9),
    ("up", "attentions", 2, 2, 8),
    ("up", "attentions", 2, 1, 7),
    ("up", "upsampler", 2, 0, 6),
    ("up", "attentions", 1, 2, 5),
    ("up", "attentions", 1, 1, 4),
    ("up", "upsampler", 1, 0, 3),
    ("up", "attentions", 0, 2, 2),
    ("up", "attentions", 0, 1, 1),
    ("up", "attentions", 0, 0, 0),
]

CUSTOM_LAYER_NAME = [
    ("down", "attentions", 0, 0, 1),
    ("down", "attentions", 0, 1, 2),
    ("down", "resnet", 0, 0, 1),
    ("down", "resnet", 0, 1, 2),
    ("down", "downsampler", 0, 2, 3),
    ("down", "attentions", 1, 0, 4),
    ("down", "attentions", 1, 1, 5),
    ("down", "resnet", 1, 0, 4),
    ("down", "resnet", 1, 1, 5),
    ("down", "downsampler", 1, 2, 6),
    ("down", "attentions", 2, 0, 7),
    ("down", "attentions", 2, 1, 8),
    ("down", "resnet", 2, 0, 7),
    ("down", "resnet", 2, 1, 8),
    ("down", "downsampler", 2, 2, 9),
    ("down", "resnet", 3, 0, 10),
    ("down", "resnet", 3, 1, 11),
    ("mid", "mid_block", 0, 0, 12),
    ("up", "resnet", 3, 2, 11),
    ("up", "resnet", 3, 1, 10),
    ("up", "resnet", 3, 0, 9),
    ("up", "upsampler", 3, 0, 9),
    ("up", "attentions", 2, 2, 8),
    ("up", "attentions", 2, 1, 7),
    ("up", "attentions", 2, 0, 6),
    ("up", "resnet", 2, 2, 8),
    ("up", "resnet", 2, 1, 7),
    ("up", "resnet", 2, 0, 6),
    ("up", "upsampler", 2, 0, 6),
    ("up", "attentions", 1, 2, 5),
    ("up", "attentions", 1, 1, 4),
    ("up", "attentions", 1, 0, 3),
    ("up", "resnet", 1, 2, 5),
    ("up", "resnet", 1, 1, 4),
    ("up", "resnet", 1, 0, 3),
    ("up", "upsampler", 1, 0, 3),
    ("up", "attentions", 0, 2, 2),
    ("up", "attentions", 0, 1, 1),
    ("up", "attentions", 0, 0, 0),
    ("up", "resnet", 0, 2, 2),
    ("up", "resnet", 0, 1, 1),
    ("up", "resnet", 0, 0, 0),
]

ORDERED_BRANCH_NAME = [
    ("down", 1),
    ("down", 2),
    ("down", 3),
    ("down", 4),
    ("down", 5),
    ("down", 6),
    ("down", 7),
    ("down", 8),
    ("down", 9),
    ("down", 10),
    ("down", 11),
    ("mid", 12),
    ("up", 11),
    ("up", 10),
    ("up", 9),
    ("up", 8),
    ("up", 7),
    ("up", 6),
    ("up", 5),
    ("up", 4),
    ("up", 3),
    ("up", 2),
    ("up", 1),
    ("up", 0),
]

SDXL_BRANCH_NAME = [
    ("down", 1),
    ("down", 2),
    ("down", 3),
    ("down", 4),
    ("down", 5),
    ("down", 6),
    ("down", 7),
    ("down", 8),
    ("mid", 9),
    ("up", 8),
    ("up", 7),
    ("up", 6),
    ("up", 5),
    ("up", 4),
    ("up", 3),
    ("up", 2),
    ("up", 1),
    ("up", 0),
]


class Attention(nn.Module):
    def __init__(self, d_model, reduction=4):
        super().__init__()
        # self.d_model = d_model
        self.kmlp = nn.Linear(d_model, d_model // reduction, bias=False)
        self.qmlp = nn.Linear(d_model, d_model // reduction, bias=False)
        self.vmlp = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k=None, v=None):
        # q: (B, N, d)
        if k is None:
            k = q

        if v is None:
            v = q

        q = self.qmlp(q)
        k = self.kmlp(k)
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / (math.sqrt(q.shape[-1]))

        attn_map = F.softmax(u, dim=-1)
        output = torch.bmm(attn_map, v)
        output = self.vmlp(output)
        return output


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            act="relu",
            zero_init=True):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        self.act = _get_activation_fn(act)

        # assert output_dim == 2
        m_list = []
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            if i == len([input_dim] + h) - 1 and zero_init:
                m = nn.Linear(n, k)
                m.weight.data[0] = torch.zeros_like(m.weight.data[0])
                m.weight.data[1] = torch.zeros_like(m.weight.data[1])
                m.bias.data[0] = torch.zeros_like(m.bias.data[0])
                m.bias.data[1] = torch.ones_like(
                    m.bias.data[1]
                )  # 0-1 initialization for last layer
                m_list.append(m)
            else:
                m_list.append(nn.Linear(n, k))

        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(m_list)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PreMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.act = _get_activation_fn(act)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        attention_type,
        self_step_attn_module_type,
        dropout=0.1,
        act="relu",
        n_step=50,
    ):
        super().__init__()
        self.self_step_attn_module_type = self_step_attn_module_type
        self.n_step = n_step
        self.attention_type = attention_type.split("-")
        self.act = _get_activation_fn(act)

        assert len(attention_type) != 0

        if "cross_step" in attention_type:
            self.cross_attn = Attention(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)

        if self_step_attn_module_type == "model_wise":
            if "pre_self_step" in attention_type:
                self.pre_self_attn = Attention(d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)

            if "pre_mlp" in attention_type:
                self.pre_linear1 = nn.Linear(d_model, d_model // 4)
                self.pre_linear2 = nn.Linear(d_model // 4, d_model)
                self.pre_dropout = nn.Dropout(dropout)
                self.pre_norm4 = nn.LayerNorm(d_model)
                self.pre_dropout4 = nn.Dropout(dropout)

            if "post_self_step" in attention_type:
                self.post_self_attn = Attention(d_model)
                self.dropout3 = nn.Dropout(dropout)
                self.norm3 = nn.LayerNorm(d_model)

            if "post_mlp" in attention_type:
                self.linear1 = nn.Linear(d_model, d_model // 4)
                self.linear2 = nn.Linear(d_model // 4, d_model)
                self.dropout = nn.Dropout(dropout)
                self.norm4 = nn.LayerNorm(d_model)
                self.dropout4 = nn.Dropout(dropout)

        elif self_step_attn_module_type == "step_wise":
            if "pre_self_step" in attention_type:
                self.pre_self_attn = _get_clones(Attention(d_model), n_step)
                self.norm1 = _get_clones(nn.LayerNorm(d_model), n_step)
                self.dropout1 = _get_clones(nn.Dropout(dropout), n_step)

            if "pre_mlp" in attention_type:
                self.pre_linear1 = _get_clones(
                    nn.Linear(d_model, d_model // 4), n_step)
                self.pre_linear2 = _get_clones(
                    nn.Linear(d_model // 4, d_model), n_step)
                self.pre_dropout = _get_clones(nn.Dropout(dropout), n_step)
                self.pre_norm4 = _get_clones(nn.LayerNorm(d_model), n_step)
                self.pre_dropout4 = _get_clones(nn.Dropout(dropout), n_step)

            if "post_self_step" in attention_type:
                self.post_self_attn = _get_clones(Attention(d_model), n_step)
                self.dropout3 = _get_clones(nn.Dropout(dropout), n_step)
                self.norm3 = _get_clones(nn.LayerNorm(d_model), n_step)

            if "post_mlp" in attention_type:
                self.linear1 = _get_clones(
                    nn.Linear(d_model, d_model // 4), n_step)
                self.linear2 = _get_clones(
                    nn.Linear(d_model // 4, d_model), n_step)
                self.dropout = _get_clones(nn.Dropout(dropout), n_step)
                self.norm4 = _get_clones(nn.LayerNorm(d_model), n_step)
                self.dropout4 = _get_clones(nn.Dropout(dropout), n_step)
        else:
            raise NotImplementedError

    def model_wise_forward(self, x):
        # query_embed: (T, N, D)
        if "pre_self_step" in self.attention_type:
            x = x + self.dropout1(self.pre_self_attn(x))
            x = self.norm1(x)

        if "pre_mlp" in self.attention_type:
            x = x + self.pre_dropout4(
                self.pre_linear2(self.pre_dropout(self.act(self.pre_linear1(x))))
            )
            x = self.pre_norm4(x)

        if "cross_step" in self.attention_type:
            x = x.permute(1, 0, 2)  # (N, T, D)
            x = x + self.dropout2(self.cross_attn(x))
            x = self.norm2(x)
            x = x.permute(1, 0, 2)

        if "post_self_step" in self.attention_type:
            x = x + self.dropout3(self.post_self_attn(x))
            x = self.norm3(x)

        if "post_mlp" in self.attention_type:
            x = x + \
                self.dropout4(self.linear2(self.dropout(self.act(self.linear1(x)))))
            x = self.norm4(x)

        return x

    def step_wise_forward(self, x):
        # query_embed: (T, N, D)
        if "pre_self_step" in self.attention_type:
            temp_xs = []
            for i in range(self.n_step):
                x_i = x[i: i + 1]
                x_i = x_i + self.dropout1[i](self.pre_self_attn[i](x_i))
                x_i = self.norm1[i](x_i)
                temp_xs.append(x_i)
            x = torch.cat(temp_xs, dim=0)

        if "pre_mlp" in self.attention_type:
            temp_xs = []
            for i in range(self.n_step):
                x_i = x[i: i + 1]
                x_i = x_i + self.pre_dropout4[i](
                    self.pre_linear2[i](
                        self.pre_dropout[i](self.act(self.pre_linear1[i](x_i)))
                    )
                )
                x_i = self.pre_norm4[i](x_i)
                temp_xs.append(x_i)
            x = torch.cat(temp_xs, dim=0)

        if "cross_step" in self.attention_type:
            x = x.permute(1, 0, 2)  # (N, T, D)
            x = x + self.dropout2(self.cross_attn(x))
            x = self.norm2(x)
            x = x.permute(1, 0, 2)

        if "post_self_step" in self.attention_type:
            temp_xs = []
            for i in range(self.n_step):
                x_i = x[i: i + 1]
                x_i = x_i + self.dropout3[i](self.post_self_attn[i](x_i))
                x_i = self.norm3[i](x_i)
                temp_xs.append(x_i)
            x = torch.cat(temp_xs, dim=0)

        if "post_mlp" in self.attention_type:
            temp_xs = []
            for i in range(self.n_step):
                x_i = x[i: i + 1]
                x_i = x_i + self.dropout4[i](
                    self.linear2[i](self.dropout[i](self.act(self.linear1[i](x_i))))
                )
                x_i = self.norm4[i](x_i)
                temp_xs.append(x_i)
            x = torch.cat(temp_xs, dim=0)

        return x

    def forward(self, x):
        # query_embed: (T, N, D)
        if self.self_step_attn_module_type == "model_wise":
            self.model_wise_forward(x)
        elif self.self_step_attn_module_type == "step_wise":
            self.step_wise_forward(x)
        else:
            raise NotImplementedError
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        attention_type,
        self_step_attn_module_type,
        dropout=0.1,
        act="relu",
        n_step=50,
    ):
        super().__init__()
        encoder_layer = EncoderLayer(
            d_model,
            attention_type,
            self_step_attn_module_type,
            dropout,
            act,
            n_step)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Pruner(nn.Module):
    def __init__(
            self,
            args,
            unet=None,
            num_inference_steps=50,
            is_sdxl_style=False):
        super().__init__()
        # ignore_blocks = args.ignore_blocks
        # if ignore_blocks is None:
        #     ignore_blocks = []  # ignore 0, 8 in some cases
        # else:
        #     ignore_blocks = [int(x) for x in ignore_blocks.split("_")]

        if unet is not None:
            self.unet = unet
        self.num_inference_steps = num_inference_steps
        # if timesteps is not None: self.timesteps = [x.item() for x in list(timesteps)]
        # [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161, 141, 121, 101, 81, 61, 41, 21, 1]

        # modules:
        self.cache_mask = None
        self.query_embed = None
        self.encoder = None
        self.use_attn = args.use_attn
        self.attention_type = (
            args.attention_type
        )  # self_step / cross_step / post_attn / post_mlp
        self.hidden_dim = args.hidden_dim
        self.encoder_layer_num = args.encoder_layer_num
        self.mlp_module_type = (
            args.mlp_module_type
        )  # block_wise / step_wise / model_wise
        self.mlp_layer_num = args.mlp_layer_num  # default 3
        self.activation_type = args.activation_type  # default relu
        self.self_step_attn_module_type = (
            args.self_step_attn_module_type
        )  # step_wise / model_wise

        self.is_sdxl_style = is_sdxl_style
        if is_sdxl_style:
            self.branch_names = SDXL_BRANCH_NAME
            self.branch_num = 9

        else:
            self.branch_names = ORDERED_BRANCH_NAME
            self.branch_num = 12

        self.model_type = args.pretrained_model_name_or_path

        self.start_branch = args.start_branch
        self.prior_gates, self.prior_gate_num = self.get_prior_gates(
            self.start_branch)

        self.filtered_branch_names = [
            x for x in self.branch_names if x not in self.prior_gates.keys()
        ]
        print("Target block names: ", self.filtered_branch_names)
        print("Prior block names: ", self.prior_gates.keys())

        self.num_gates = len(self.branch_names) - self.prior_gate_num
        self.inject_cache_mask_module()

        # intermediate variables
        self.all_timestep_logits = None  # prediction logits
        self.all_timestep_gates = None  # 0-1 gates
        self.all_timestep_original_gates = None  # 0-1 gates
        self.all_timestep_scores = None  # normalized prediction scores

        (
            self.sub_block_flops,
            self.sub_block_flops_ratios,
            self.branch_flops,
            self.branch_flops_ratios,
        ) = self.get_flops()
        self.cascade_nodes = self.get_cascade_nodes()
        self.cascade_flops, self.cascade_ratios = self.get_cascade_flops()

    def get_prior_gates(self, start_branch):
        prior_gates = {}
        prior_gate_num = 0
        for blocktype, branch_i in self.branch_names:
            if branch_i >= start_branch:
                continue
            prior_gates[(blocktype, branch_i)] = torch.tensor(
                [0, 1], requires_grad=False
            )
            prior_gate_num += 1
        return prior_gates, prior_gate_num

    def get_flops(self, inplace=False):
        # example_inputs = {
        #     'sample': torch.randn(1, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size).to(
        #         dtype=self.unet.dtype),
        #     'timestep': torch.Tensor([981]).long(),
        #     'encoder_hidden_states': torch.randn(1, 77, 768).to(dtype=self.unet.dtype)
        # }

        unet = self.unet
        if self.is_sdxl_style:
            example_inputs = {
                "sample": torch.randn(
                    1, unet.in_channels, unet.sample_size, unet.sample_size
                ),
                "timestep": torch.Tensor([981]).long(),
                "encoder_hidden_states": torch.randn(1, 77, 2048),
                "added_cond_kwargs": {
                    "text_embeds": torch.randn(1, 1280),
                    "time_ids": torch.randn((1, 6)),
                },
                # 'text_embeds': torch.randn(1, 77, 768)
            }
        else:
            example_inputs = {
                "sample": torch.randn(
                    1, unet.in_channels, unet.sample_size, unet.sample_size
                ),
                "timestep": torch.Tensor([981]).long(),
                "encoder_hidden_states": (
                    torch.randn(1, 77, 1024)
                    if self.model_type == "sd2.1"
                    else torch.randn(1, 77, 768)
                ),
            }
        # model = copy.deepcopy(self.unet)
        results = count_ops_and_params(
            unet.float(),
            inplace=inplace,
            example_inputs=example_inputs,
            layer_wise=False,
            block_wise=False,
            branch_wise=True,
        )
        (
            macs,
            nparams,
            block_flops,
            block_params,
            block_flops_ratios,
            sub_block_flops,
            sub_block_flops_ratios,
            branch_flops,
            branch_flops_ratios,
        ) = results

        del example_inputs
        torch.cuda.empty_cache()

        # block_flops = {self.block_name_mapping.get(k): v for k, v in block_flops.items()}
        # block_params = {self.block_name_mapping.get(k): v for k, v in block_params.items()}
        # block_flops_ratios = {self.block_name_mapping.get(k): v for k, v in block_flops_ratios.items()}

        print("#Params: {:.4f} M".format(nparams / 1e6))
        print("#MACs: {:.4f} G".format(macs / 1e9))
        return (
            sub_block_flops,
            sub_block_flops_ratios,
            branch_flops,
            branch_flops_ratios,
        )

    def get_average_flops(self):
        all_timesteps_flops = 0.0
        unet = self.unet
        if self.is_sdxl_style:
            example_inputs = {
                "sample": torch.randn(
                    1, unet.in_channels, unet.sample_size, unet.sample_size
                ),
                "timestep": torch.Tensor([981]).long(),
                "encoder_hidden_states": torch.randn(1, 77, 2048),
                "added_cond_kwargs": {
                    "text_embeds": torch.randn(1, 1280),
                    "time_ids": torch.randn((1, 6)),
                },
                "return_dict": False,
                # 'text_embeds': torch.randn(1, 77, 768)
            }
        else:
            example_inputs = {
                "sample": torch.randn(
                    1, unet.in_channels, unet.sample_size, unet.sample_size
                ),
                "timestep": torch.Tensor([981]).long(),
                "encoder_hidden_states": (
                    torch.randn(1, 77, 1024)
                    if self.model_type == "sd2.1"
                    else torch.randn(1, 77, 768)
                ),
                "return_dict": False,
            }
        dummy_prev_features = unet.cpu().float()(**example_inputs)[1]

        for t in tqdm(range(self.num_inference_steps)):
            if self.is_sdxl_style:
                example_inputs = {
                    "sample": torch.randn(
                        1,
                        self.unet.in_channels,
                        self.unet.sample_size,
                        self.unet.sample_size,
                    ),
                    "timestep": torch.Tensor([t]).long(),
                    "replicate_prv_feature": dummy_prev_features if t > 0 else None,
                    "gates": self.all_timestep_gates[t - 1] if t > 0 else None,
                    "is_gates_train": False,
                    "encoder_hidden_states": torch.randn(1, 77, 2048),
                    "added_cond_kwargs": {
                        "text_embeds": torch.randn(1, 1280),
                        "time_ids": torch.randn((1, 6)),
                    },
                }
            else:
                example_inputs = {
                    "sample": torch.randn(
                        1,
                        self.unet.in_channels,
                        self.unet.sample_size,
                        self.unet.sample_size,
                    ),
                    "timestep": torch.Tensor([t]).long(),
                    "replicate_prv_feature": dummy_prev_features if t > 0 else None,
                    "gates": self.all_timestep_gates[t - 1] if t > 0 else None,
                    "is_gates_train": False,
                    "encoder_hidden_states": (
                        torch.randn(1, 77, 1024)
                        if self.model_type == "sd2.1"
                        else torch.randn(1, 77, 768)
                    ),
                }
            # model = copy.deepcopy(self.unet)
            results = count_ops_and_params(
                unet.float(),
                inplace=True,
                example_inputs=example_inputs,
                layer_wise=False,
                block_wise=False,
                branch_wise=True,
            )
            macs = results[0]

            del example_inputs
            torch.cuda.empty_cache()
            all_timesteps_flops += macs

        print(
            "#Average MACs: {:.4f} G".format(
                all_timesteps_flops / self.num_inference_steps / 1e9
            )
        )
        return all_timesteps_flops / self.num_inference_steps / 1e9

    def enable(self, unet=None):
        assert self.unet is not None
        self.reset_states()
        self.wrap_modules()

    def is_reuse(self, query):
        return query[0] == 1

    def disable(self):
        self.unwrap_modules()
        self.reset_states()

    def set_params(self, cache_gate_list: list):
        # cache_gate_list: length = N_timesteps, e.g., [[0, 1, 1, 0, 0, 1, 0],
        # ...]. 0 means reuse
        device = self.unet.device
        t_num = self.num_inference_steps - 1  # ignore first step

        assert len(cache_gate_list) == t_num
        assert len(cache_gate_list[0]) == len(self.branch_names)

        all_timestep_gates = [
            {
                block_name: torch.Tensor([0, 1]).to(device)
                for block_name in self.branch_names
            }
            for _ in range(t_num)
        ]

        for i in range(len(cache_gate_list)):
            for j in range(len(cache_gate_list[0])):
                if cache_gate_list[i][j] == 0:
                    block_name = self.branch_names[j]
                    all_timestep_gates[i][block_name] = torch.Tensor(
                        [1, 0]).to(device)

            all_timestep_gates[i] = self.update_query_status(
                all_timestep_gates[i]
            )  # update query status according to the topology

        self.all_timestep_gates = all_timestep_gates

    def is_skip_step(self):
        self.start_timestep = (
            self.cur_timestep if self.start_timestep is None else self.start_timestep
        )  # For some pipeline that the first timestep != 0
        if self.cur_timestep == 0:
            return False
        else:
            return True

    def custom_gumbel_softmax(self, logits, dim=-1):
        y_soft = logits.softmax(dim=dim)
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def queries_forward(self, query_embed, time_id):
        # 1. Set prior logits
        original_query_gate = copy.deepcopy(self.prior_gates)
        query_logits = copy.deepcopy(self.prior_gates)
        query_scores = copy.deepcopy(self.prior_gates)

        for _, (blocktype, branch_i) in enumerate(self.branch_names):
            if (blocktype, branch_i) in self.prior_gates.keys():
                continue
            i = self.filtered_branch_names.index((blocktype, branch_i))
            query = query_embed[time_id][i]
            if self.mlp_module_type == "block_wise":
                query = self.cache_mask[time_id][i](query)  # vector (0, 1)
            elif self.mlp_module_type == "partial_step_wise":
                query = self.cache_mask[time_id][0](query)
                query = self.cache_mask[time_id][1][i](query)
            elif self.mlp_module_type == "partial_model_wise":
                query = self.cache_mask[0](query)
                query = self.cache_mask[1][time_id][i](query)
            elif self.mlp_module_type == "step_wise":
                query = self.cache_mask[time_id](query)
            elif self.mlp_module_type == "model_wise":
                query = self.cache_mask(query)
            else:
                raise NotImplementedError
            query_logits[(blocktype, branch_i)] = query

            if self.training:
                gate = self.custom_gumbel_softmax(query)
            else:
                max_index = torch.argmax(query, dim=-1)
                gate = torch.zeros_like(
                    query, dtype=query.dtype).to(
                    query.device)
                gate[max_index] = 1.0

            original_query_gate[(blocktype, branch_i)] = gate

        updated_query_gate = self.update_query_status(original_query_gate)

        for k, v in query_logits.items():
            if k in self.prior_gates.keys():
                continue
            query_scores[k] = F.softmax(v)

        return updated_query_gate, original_query_gate, query_logits, query_scores

    def update_query_status(self, query_gates, query_logits=None):
        # 1. up
        for branch_i in range(self.branch_num):
            gate = query_gates[("up", branch_i)]
            if self.is_reuse(gate):
                if branch_i + 1 < self.branch_num:
                    pre_layer = ("up", branch_i + 1)
                else:
                    pre_layer = ("mid", branch_i + 1)
                if not self.is_reuse(query_gates[pre_layer]):
                    query_gates[pre_layer] = query_gates[pre_layer] * \
                        gate + gate
                else:
                    query_gates[pre_layer] = query_gates[pre_layer] * gate

                if branch_i == 0:
                    continue

                lateral_layer = ("down", branch_i)
                if not self.is_reuse(query_gates[lateral_layer]):
                    query_gates[lateral_layer] = (
                        query_gates[lateral_layer] * gate + gate
                    )
                else:
                    query_gates[lateral_layer] = query_gates[lateral_layer] * gate

        return query_gates

    def get_cascade_nodes(self):
        cascade_nodes = {}

        for branch_i in range(self.branch_num):
            target_name = ("up", branch_i)
            if branch_i == 0:
                cascade_nodes[target_name] = [
                    ("down", i) for i in range(1, self.branch_num)
                ]
            else:
                cascade_nodes[target_name] = [
                    ("down", i) for i in range(branch_i, self.branch_num)
                ]
            cascade_nodes[target_name].extend(
                [("up", i) for i in range(branch_i, self.branch_num)]
            )
            cascade_nodes[target_name].append(("mid", self.branch_num))

        cascade_nodes[("mid", self.branch_num)] = [("mid", self.branch_num)]
        for branch_i in range(1, self.branch_num):
            cascade_nodes[("down", branch_i)] = [("down", branch_i)]

        # print("Cascade Nodes: ", cascade_nodes)
        return cascade_nodes

    @torch.no_grad()
    def evaluate_sparsity(self, verbose=True):
        n_time = self.num_inference_steps - 1
        actual_ops = 1.0  # start from 0 to count the first step
        with torch.no_grad():
            all_timestep_gates = self.all_queries_forward(update=False)[0]
            for t in range(n_time):
                updated_query_gate = all_timestep_gates[t]
                # compute sparsity
                for k, v in updated_query_gate.items():
                    if not self.is_reuse(v):
                        actual_ops += self.branch_flops_ratios[k]

                # for ignore_block in self.ignore_block_names:
                #     actual_ops += self.block_flops_ratios[ignore_block]

        sparsity = actual_ops / self.num_inference_steps

        if verbose:
            print(f"Evaluating Sparsity: {sparsity}.")
        return sparsity

    def sparse_lossv2(self, smooth=False):
        # get block cascade nodes
        # if block is reuse status, update the gates;
        # if not, update the gates
        all_time_flops_ratio = []
        cascade_ratios = self.cascade_ratios
        for i, gates in enumerate(
            self.all_timestep_original_gates
        ):  # 0 for skip, 1 for keep
            cur_time_flops_ratio = 0.0
            # for block_name in self.branch_names:
            for block_name in self.filtered_branch_names:
                if smooth:
                    cur_time_flops_ratio += gates[block_name][1] * np.sqrt(
                        cascade_ratios[block_name]
                    )
                else:
                    cur_time_flops_ratio += (
                        gates[block_name][1] * cascade_ratios[block_name]
                    )
            all_time_flops_ratio.append(cur_time_flops_ratio)
        return sum(all_time_flops_ratio) / len(all_time_flops_ratio)

    def get_gate_index(self, blocktype, branch_i):
        if blocktype == "down" or blocktype == "mid":
            return branch_i - 1 - self.start_branch
        else:
            return (
                branch_i + self.branch_num - self.start_branch
            )  # 12 branches in total

    def inject_cache_mask_module(self):
        """
        inject lora into model, and returns lora parameter groups.
        Inspired from https://github.com/cloneofsimo/lora/blob/d84074b3e3496f1cfa8a3f49b8b9972ef463b483/lora_diffusion/lora.py
        """
        num_timesteps = self.num_inference_steps - 1

        all_step_cache_masks = []
        self.query_embed = nn.Embedding(
            num_timesteps * self.num_gates, self.hidden_dim)

        if self.use_attn:
            self.encoder = Encoder(
                self.hidden_dim,
                self.encoder_layer_num,
                self.attention_type,
                self.self_step_attn_module_type,
                act=self.activation_type,
                n_step=num_timesteps,
            )
        if self.mlp_module_type == "block_wise":
            for i in range(num_timesteps):
                mlp = MLP(
                    self.hidden_dim,
                    self.hidden_dim // 4,
                    2,
                    num_layers=self.mlp_layer_num,
                    act=self.activation_type,
                )
                all_step_cache_masks.append(_get_clones(mlp, self.num_gates))
            self.cache_mask = nn.ModuleList(all_step_cache_masks)
        elif self.mlp_module_type == "partial_model_wise":
            pre_mlp = PreMLP(
                self.hidden_dim,
                self.hidden_dim // 4,
                self.hidden_dim // 4,
                num_layers=self.mlp_layer_num - 1,
                act=self.activation_type,
            )
            for i in range(num_timesteps):
                head_mlp = MLP(
                    self.hidden_dim // 4,
                    self.hidden_dim // 4,
                    2,
                    num_layers=1,
                    act=self.activation_type,
                    zero_init=True,
                )
                mask_for_one_time_step = _get_clones(head_mlp, self.num_gates)
                all_step_cache_masks.append(mask_for_one_time_step)
            self.cache_mask = nn.ModuleList(
                [pre_mlp, nn.ModuleList(all_step_cache_masks)]
            )
        elif self.mlp_module_type == "partial_step_wise":
            for i in range(num_timesteps):
                pre_mlp = PreMLP(
                    self.hidden_dim,
                    self.hidden_dim // 4,
                    self.hidden_dim // 4,
                    num_layers=self.mlp_layer_num - 1,
                    act=self.activation_type,
                )
                head_mlp = MLP(
                    self.hidden_dim // 4,
                    self.hidden_dim // 4,
                    2,
                    num_layers=1,
                    act=self.activation_type,
                    zero_init=True,
                )
                mask_for_one_time_step = _get_clones(head_mlp, self.num_gates)
                all_step_cache_masks.append(
                    nn.ModuleList([pre_mlp, mask_for_one_time_step])
                )
            self.cache_mask = nn.ModuleList(all_step_cache_masks)
        elif self.mlp_module_type == "step_wise":
            for i in range(num_timesteps):
                mlp = MLP(
                    self.hidden_dim,
                    self.hidden_dim // 4,
                    2,
                    num_layers=self.mlp_layer_num,
                    act=self.activation_type,
                )
                all_step_cache_masks.append(mlp)
            self.cache_mask = nn.ModuleList(all_step_cache_masks)
        elif self.mlp_module_type == "model_wise":
            self.cache_mask = MLP(
                self.hidden_dim,
                self.hidden_dim // 4,
                2,
                num_layers=self.mlp_layer_num,
                act=self.activation_type,
            )

    def save_state_dict(self, path):
        if self.encoder is not None:
            state_dict = {
                "query_embed": self.query_embed.state_dict(),
                "cache_mask": self.cache_mask.state_dict(),
                "encoder": self.encoder.state_dict(),
            }
        else:
            state_dict = {
                "query_embed": self.query_embed.state_dict(),
                "cache_mask": self.cache_mask.state_dict(),
            }
        torch.save(state_dict, path)

    def load_state_dict(self, state_dict, strict: bool = True):
        # state_dict = torch.load(state_dict_path)
        try:
            self.query_embed.load_state_dict(state_dict["query_embed"])
        except BaseException:
            weight = torch.zeros_like(self.query_embed.weight.data)
            weight = weight.reshape(
                self.num_inference_steps - 1, -1, weight.shape[-1])
            for i in range(self.num_inference_steps - 1):
                weight[i] = state_dict["query_embed"][f"{i}.weight"]
            self.query_embed.weight.data = weight.reshape(-1, weight.shape[-1])

        self.cache_mask.load_state_dict(state_dict["cache_mask"])

        if "encoder" in state_dict.keys() and self.encoder is not None:
            self.encoder.load_state_dict(state_dict["encoder"])
        else:
            self.encoder = None
            print("Missing Key: encoder.")
        # self.timesteps = state_dict["timesteps"]

    def all_queries_forward(self, update=True):
        # forward queries of all time steps in a once
        all_timestep_gates = []
        all_timestep_original_gates = []
        all_timestep_logits = []
        all_timestep_scores = []

        n_timestep = self.num_inference_steps - 1
        query_embed = self.query_embed.weight
        query_embed = query_embed.reshape(
            n_timestep, self.num_gates, self.hidden_dim)

        if self.encoder is not None and self.use_attn:
            query_embed = self.encoder(query_embed)

        for i in range(n_timestep):
            query_gate, original_query_gate, query_logits, query_scores = (
                self.queries_forward(query_embed, i)
            )  # cur timestep  0-50
            all_timestep_gates.append(query_gate)
            all_timestep_original_gates.append(original_query_gate)
            all_timestep_logits.append(query_logits)
            all_timestep_scores.append(query_scores)

        if update:
            self.all_timestep_gates = all_timestep_gates
            self.all_timestep_original_gates = all_timestep_original_gates
            self.all_timestep_logits = all_timestep_logits
            self.all_timestep_scores = all_timestep_scores

        return (
            all_timestep_gates,
            all_timestep_original_gates,
            all_timestep_logits,
            all_timestep_scores,
        )

    # @staticmethod
    def get_cascade_flops(self):
        cascade_nodes = self.get_cascade_nodes()
        cascade_flops = {}
        cascade_ratios = {}
        for block_name in cascade_nodes.keys():
            cascade_ratios[block_name] = sum(
                [self.branch_flops_ratios[x] for x in cascade_nodes[block_name]]
            )
            cascade_flops[block_name] = sum(
                [self.branch_flops[x] for x in cascade_nodes[block_name]]
            )
        return cascade_flops, cascade_ratios

    @torch.no_grad()
    def prune(self, expected_ratio, prune_threshold=0.95):
        # expected_ratio: ratios for pruned flops per time steps
        # original_block_flops, original_block_ratios = self.branch_flops, self.branch_flops_ratios
        cascade_nodes = self.cascade_nodes
        device = self.unet.device
        t_num = self.num_inference_steps - 1
        all_timestep_gates = [
            {
                block_name: torch.Tensor([0, 1]).to(device)
                for block_name in self.branch_names
            }
            for _ in range(t_num)
        ]
        all_timestep_scores = self.all_queries_forward(update=False)[3]

        actual_pruned_ratio = 0.0
        while actual_pruned_ratio < expected_ratio:
            for branch_name in self.filtered_branch_names[::-1]:
                all_timestep_target_scores = []
                all_timestep_target_gates = []
                for t in range(t_num):
                    target_scores = all_timestep_scores[t][branch_name]
                    target_gates = all_timestep_gates[t][branch_name]
                    all_timestep_target_scores.append(target_scores)
                    all_timestep_target_gates.append(target_gates)

                sorted_indices = sorted(
                    range(len(all_timestep_target_scores)),
                    key=lambda k: all_timestep_target_scores[k][0],
                    reverse=True,
                )
                filtered_indices = list(
                    filter(
                        lambda x: not self.is_reuse(all_timestep_gates[x][branch_name])
                        and all_timestep_target_scores[x][0] > prune_threshold,
                        sorted_indices,
                    )
                )
                threshold_based_prune_num = len(filtered_indices)
                pruned_num = threshold_based_prune_num
                pruned_indices = filtered_indices[:pruned_num]

                # update all timestep gates
                cur_pruned_ratios = 0.0
                for pruned_t in pruned_indices:
                    cur_time_pruned_ratios = 0.0
                    # 1. estimate the ratios will be pruned
                    for cascade_block_name in cascade_nodes[branch_name]:
                        if (
                            all_timestep_gates[pruned_t][cascade_block_name]
                            == torch.Tensor([0, 1]).to(device)
                        ).all():
                            cur_time_pruned_ratios += self.branch_flops_ratios[
                                cascade_block_name
                            ]

                    # normalized by actual len(self.timesteps)
                    if (
                        actual_pruned_ratio
                        + (cur_pruned_ratios + cur_time_pruned_ratios)
                        / self.num_inference_steps
                    ) >= expected_ratio:
                        break

                    # 2. pruned the specified nodes (set gates to reuse)
                    for cascade_block_name in cascade_nodes[branch_name]:
                        if (
                            all_timestep_gates[pruned_t][cascade_block_name]
                            == torch.Tensor([0, 1]).to(device)
                        ).all():
                            all_timestep_gates[pruned_t][cascade_block_name] = (
                                torch.Tensor([1, 0]).to(device))

                    # 3. count the pruned ratios
                    cur_pruned_ratios += cur_time_pruned_ratios

                cur_pruned_ratios = cur_pruned_ratios / self.num_inference_steps
                actual_pruned_ratio += cur_pruned_ratios
                if actual_pruned_ratio >= expected_ratio:
                    break

            prune_threshold = (
                prune_threshold - 0.05
            )  # decrease threshold for more blocks to prune
            if prune_threshold < 0:
                break

        print(
            f"Prune done. End with pruned ratios {actual_pruned_ratio} and threshold {prune_threshold}."
        )
        self.all_timestep_gates = all_timestep_gates
        return all_timestep_gates


"""
# example for using set_params:
import diffusers
from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
)
import random

random_numbers = [[random.randint(0, 1) for _ in range(9)] for _ in range(50)]
# print(random_numbers)
noise_scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler", cache_dir="/data/zhuhaowei/huggingface")
noise_scheduler.set_timesteps(num_inference_steps=50)
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", cache_dir="/data/zhuhaowei/huggingface")
helper = UNetHelper(unet=unet, timesteps=noise_scheduler.timesteps, use_attn=3)
helper.set_params(random_numbers)
helper.enable()
helper.eval()
helper.get_average_flops()
"""

# query_embed[time_id].weight[block_name]
# query_embed = nn.ModuleList(nn.Embedding(50, self.hidden_dim) for _ in range(num_timesteps))
# query_embed = nn.Embedding(50 * 10, 512)
# query_embed = query_embed.weight.reshape(50, 10, 512)
# encoder = Encoder(512, 1)
# query_embed = encoder(query_embed)
# mlp = MLP(512, 512 // 4, 2, 3)
# # print(mlp.layers[-1].weight)
# print(mlp(query_embed))
