import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
import pickle
import math
from flops import count_ops_and_params
from tqdm import tqdm
from typing import Optional, List


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


class Node:
    def __init__(self, name_predecessors, name_successors):
        self.name_predecessors = name_predecessors
        self.name_successors = name_successors


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


def recursive_multipy(data, scalar_tensor, detach=False):
    if isinstance(data, torch.Tensor):
        if detach:
            return data.detach() * scalar_tensor
        return data * scalar_tensor
    elif isinstance(data, tuple):
        return tuple(recursive_multipy(item, scalar_tensor) for item in data)
    else:
        return data


def recursive_add(tuple1, tuple2):
    if isinstance(tuple1, torch.Tensor) and isinstance(tuple2, torch.Tensor):
        return tuple1 + tuple2
    elif isinstance(tuple1, tuple) and isinstance(tuple2, tuple):
        return tuple(
            recursive_add(item1, item2) for item1, item2 in zip(tuple1, tuple2)
        )
    elif isinstance(tuple1, list) and isinstance(tuple2, list):
        return list(recursive_add(item1, item2)
                    for item1, item2 in zip(tuple1, tuple2))
    else:
        print(type(tuple1), type(tuple2))
        raise ValueError("Input tuples must have the same structure")


class UNetHelper(nn.Module):
    def __init__(self, args, unet=None, timesteps=None):
        super().__init__()
        ignore_blocks = args.ignore_blocks
        if ignore_blocks is None:
            ignore_blocks = []  # ignore 0, 8 in some cases
        else:
            ignore_blocks = [int(x) for x in ignore_blocks.split("_")]
        if unet is not None:
            self.unet = unet
        if timesteps is not None:
            self.timesteps = sorted(
                list(set([x.item() for x in list(timesteps)])), reverse=True
            )
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

        self.model_type = args.pretrained_model_name_or_path

        self.block_names = [
            ("down", "block", 0, 0),
            ("down", "block", 1, 0),
            ("down", "block", 2, 0),
            ("down", "block", 3, 0),
            ("mid", "mid_block", 0, 0),
            ("up", "block", 3, 0),
            ("up", "block", 2, 0),
            ("up", "block", 1, 0),
            ("up", "block", 0, 0),
        ]

        self.ignore_block_names = [self.block_names[i] for i in ignore_blocks]
        if len(ignore_blocks) > 0:
            self.block_names = [
                x for i, x in enumerate(
                    self.block_names) if i not in ignore_blocks]
            print("Filter block names: ", self.ignore_block_names)

        self.block_name_mapping = {
            "down_blocks.0": ("down", "block", 0, 0),
            "down_blocks.1": ("down", "block", 1, 0),
            "down_blocks.2": ("down", "block", 2, 0),
            "down_blocks.3": ("down", "block", 3, 0),
            "mid_block": ("mid", "mid_block", 0, 0),
            "up_blocks.0": ("up", "block", 3, 0),
            "up_blocks.1": ("up", "block", 2, 0),
            "up_blocks.2": ("up", "block", 1, 0),
            "up_blocks.3": ("up", "block", 0, 0),
        }
        """
            Block mid_block:                       MACs = 6.0285 G, Params = 97.0381 M, MACs% = 1.78
            Block up_blocks.0:                     MACs = 12.9049 G, Params = 162.2413 M, MACs% = 3.81
            Block up_blocks.1:                     MACs = 75.1374 G, Params = 258.3309 M, MACs% = 22.18
            Block up_blocks.2:                     MACs = 79.1231 G, Params = 71.4131 M, MACs% = 23.35
            Block up_blocks.3:                     MACs = 66.0427 G, Params = 18.8106 M, MACs% = 19.49
            Block down_blocks.0:                   MACs = 32.9392 G, Params = 10.5245 M, MACs% = 9.72
            Block down_blocks.1:                   MACs = 31.3169 G, Params = 36.8186 M, MACs% = 9.24
            Block down_blocks.2:                   MACs = 31.4594 G, Params = 139.9923 M, MACs% = 9.28
            Block down_blocks.3:                   MACs = 3.7791 G, Params = 62.2771 M, MACs% = 1.12
            #Params: 859.5210 M
            #MACs: 338.8316 G
        """

        self.inject_cache_mask_module()

        # intermediate variables
        self.query_dict = {}
        self.all_timestep_logits = None  # prediction logits
        self.all_timestep_gates = None  # 0-1 gates
        self.all_timestep_original_gates = None  # 0-1 gates
        self.all_timestep_scores = None  # normalized prediction scores

        self.topology_dict = {}
        with open("topology.pkl", "rb") as f:
            nodes = pickle.load(f)
            self.topology_dict = nodes

        self.block_flops, self.block_flops_ratios = self.get_flops()

    def get_flops(self, inplace=False):
        unet = self.unet
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
            unet,
            inplace=inplace,
            example_inputs=example_inputs,
            layer_wise=False,
            block_wise=True,
            branch_wise=False,
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

        block_flops = {
            self.block_name_mapping.get(
                k,
                k): v for k,
            v in block_flops.items()}
        block_params = {
            self.block_name_mapping.get(
                k,
                k): v for k,
            v in block_params.items()}
        block_flops_ratios = {self.block_name_mapping.get(
            k, k): v for k, v in block_flops_ratios.items()}

        print("#Params: {:.4f} M".format(nparams / 1e6))
        print("#MACs: {:.4f} G".format(macs / 1e9))
        return block_flops, block_flops_ratios

    def get_average_flops(self, unet=None):
        all_timesteps_flops = 0.0
        if unet is None:
            unet = self.unet
        for t in tqdm(self.timesteps):
            example_inputs = {
                "sample": torch.randn(
                    1, unet.in_channels, unet.sample_size, unet.sample_size
                ),
                "timestep": torch.Tensor([t]).long(),
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
                all_timesteps_flops / len(self.timesteps) / 1e9
            )
        )
        return all_timesteps_flops / len(self.timesteps) / 1e9

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
        t_num = len(self.timesteps) - 1

        print(t_num)
        assert len(cache_gate_list) == t_num
        assert len(cache_gate_list[0]) == len(self.block_names)

        all_timestep_gates = [
            {
                block_name: torch.Tensor([0, 1]).to(device)
                for block_name in self.block_names
            }
            for _ in range(t_num)
        ]

        for i in range(len(cache_gate_list)):
            for j in range(len(cache_gate_list[0])):
                if cache_gate_list[i][j] == 0:
                    block_name = self.block_names[j]
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

    def is_enter_position(self, block_i, layer_i):
        return (
            block_i == self.params["cache_block_id"]
            and layer_i == self.params["cache_layer_id"]
        )

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.unet.forward

        def wrapped_forward(*args, **kwargs):
            try:
                # pipe.scheduler.timesteps from 981 - to 1; cur_timestep from 0
                # to 50
                self.cur_timestep = self.timesteps.index(args[1].item())
            except BaseException:
                self.cur_timestep = self.timesteps.index(
                    kwargs["timestep"].item())

            result = self.function_dict["unet_forward"](*args, **kwargs)
            return result

        self.unet.forward = wrapped_forward

    def ste(self, logits, dim=-1):
        y_soft = logits.softmax(dim=dim)
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def queries_forward(self, query_embed, time_id):
        original_query_gate = {}
        query_logits = {}

        for i, block_name in enumerate(self.block_names):
            query = query_embed[time_id][i]
            if self.mlp_module_type == "block_wise":
                query = self.cache_mask[time_id][i]["mlp"](
                    query)  # vector (0, 1)
            elif self.mlp_module_type == "partial_step_wise":
                query = self.cache_mask[time_id][0](query)
                query = self.cache_mask[time_id][1][i](query)

            elif self.mlp_module_type == "step_wise":
                query = self.cache_mask[time_id](query)
            elif self.mlp_module_type == "model_wise":
                query = self.cache_mask(query)
            else:
                raise NotImplementedError

            query_logits[block_name] = query

            if self.training:
                gate = self.ste(query)
            else:
                max_index = torch.argmax(query, dim=-1)  #
                gate = torch.zeros_like(
                    query, dtype=query.dtype).to(
                    query.device)
                gate[max_index] = 1.0

            original_query_gate[block_name] = gate

        updated_query_gate = self.update_query_status(original_query_gate)
        query_scores = {k: F.softmax(v) for k, v in query_logits.items()}
        return updated_query_gate, original_query_gate, query_logits, query_scores

    def update_query_status(self, query_gates, query_logits=None):
        topology = copy.deepcopy(self.topology_dict)
        last_block_name = ("up", "block", 0, 0)

        for block_name in self.block_names[::-1]:  # from top to bottom
            predecessors = topology[block_name].name_predecessors
            if (self.is_reuse(query_gates[block_name]) and len(
                    predecessors) > 0):  # it means reuse cache feature
                topology[block_name].name_predecessors = []
                for pre in predecessors:
                    # remove the successors of the predecessors of current node
                    removed_successors = [
                        x for x in topology[pre].name_successors if x != block_name]
                    topology[pre].name_successors = removed_successors

                    if len(removed_successors) == 0 and pre != last_block_name:

                        if not self.is_reuse(query_gates[pre]):
                            query_gates[pre] = (
                                query_gates[pre] * query_gates[block_name]
                                + query_gates[block_name]
                            )
                        else:
                            query_gates[pre] = (
                                query_gates[pre] * query_gates[block_name]
                            )

        if query_logits is not None:
            return query_gates, query_logits

        return query_gates

    def get_cascade_nodes(self):
        last_block_name = ("up", "block", 0, 0)

        cascade_nodes = {}

        for target_block_name in self.block_names[::-1]:
            topology = copy.deepcopy(self.topology_dict)
            cascade_nodes[target_block_name] = [target_block_name]
            for block_name in self.block_names[::-1]:  # from top to bottom
                predecessors = topology[block_name].name_predecessors
                if len(predecessors) > 0:
                    if block_name == target_block_name or (
                        len(topology[block_name].name_successors) == 0
                        and block_name != last_block_name
                    ):
                        topology[block_name].name_predecessors = []
                        if block_name not in cascade_nodes[target_block_name]:
                            cascade_nodes[target_block_name].append(block_name)
                        for pre in predecessors:
                            # remove the successors of the predecessors of
                            # current node
                            removed_successors = [
                                x
                                for x in topology[pre].name_successors
                                if x != block_name
                            ]
                            topology[pre].name_successors = removed_successors

                            if len(
                                    removed_successors) == 0 and pre != last_block_name:
                                cascade_nodes[target_block_name].append(pre)

        return cascade_nodes

    @torch.no_grad()
    def evaluate_sparsity(self, verbose=True):
        # n_time = len(self.timesteps) - 1
        n_time = len(self.timesteps)
        actual_ops = 0
        with torch.no_grad():
            all_timestep_gates = self.all_queries_forward(update=False)[0]
            for t in range(n_time):
                updated_query_gate = all_timestep_gates[t]
                # compute sparsity
                for k, v in updated_query_gate.items():
                    if not self.is_reuse(v):
                        actual_ops += self.block_flops_ratios[k]

                for ignore_block in self.ignore_block_names:
                    actual_ops += self.block_flops_ratios[ignore_block]

        sparsity = actual_ops / len(self.timesteps)

        if verbose:
            print(f"Evaluating Sparsity: {sparsity}.")
        return sparsity

    def sparse_lossv2(self):
        # get block cascade nodes
        # if block is reuse status, update the gates;
        # if not, update the gates
        all_time_flops_ratio = []
        cascade_ratios = self.get_cascade_flops()[1]
        for i, gates in enumerate(
            self.all_timestep_original_gates
        ):  # 0 for skip, 1 for keep
            cur_time_flops_ratio = 0.0
            for block_name in self.block_names:
                cur_time_flops_ratio += (
                    gates[block_name][1] * cascade_ratios[block_name]
                )
            all_time_flops_ratio.append(cur_time_flops_ratio)
        return sum(all_time_flops_ratio) / len(all_time_flops_ratio)

    def sparse_loss(self):
        all_time_flops_ratio = []
        for i, gates in enumerate(
            self.all_timestep_original_gates
        ):  # 0 for skip, 1 for keep
            cur_time_flops_ratio = 0.0
            for block_name in self.block_names:
                cur_time_flops_ratio += gates[block_name][1]
            all_time_flops_ratio.append(cur_time_flops_ratio)
        return sum(all_time_flops_ratio) / len(all_time_flops_ratio)

    def wrap_block_forward(
            self,
            block,
            block_name,
            block_i,
            layer_i,
            blocktype="down"):
        if (blocktype, block_name, block_i,
                layer_i) in self.ignore_block_names:
            return

        self.function_dict[(blocktype, block_name, block_i,
                            layer_i)] = block.forward

        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step()
            if skip:
                gate_name = (
                    (blocktype, "mid_block", block_i, 0)
                    if blocktype == "mid"
                    else (blocktype, "block", block_i, 0)
                )
                # gate = self.all_timestep_gates[self.cur_timestep - 1][gate_name]
                gate = self.all_timestep_gates[self.cur_timestep][gate_name]
                if not self.training:
                    if self.is_reuse(gate):
                        return self.cached_output[
                            (blocktype, block_name, block_i, layer_i)
                        ]
                    else:
                        result = self.function_dict[
                            (blocktype, block_name, block_i, layer_i)
                        ](*args, **kwargs)
                        self.cached_output[
                            (blocktype, block_name, block_i, layer_i)
                        ] = result
                        return result
                else:
                    cur_output = self.function_dict[
                        (blocktype, block_name, block_i, layer_i)
                    ](*args, **kwargs)
                    cached_output = self.cached_output[
                        (blocktype, block_name, block_i, layer_i)
                    ]
                    result = recursive_add(
                        recursive_multipy(cached_output, gate[0]),
                        recursive_multipy(cur_output, gate[1]),
                    )

                    self.cached_output[(blocktype, block_name, block_i, layer_i)] = (
                        result)
            else:
                result = self.function_dict[(blocktype, block_name, block_i, layer_i)](
                    *args, **kwargs)
                self.cached_output[(blocktype, block_name,
                                    block_i, layer_i)] = result
            return result

        block.forward = wrapped_forward

    def wrap_modules(self):
        # 1. wrap unet forward
        self.wrap_unet_forward()
        # 2. wrap downblock forward
        for block_i, block in enumerate(self.unet.down_blocks):
            self.wrap_block_forward(
                block, "block", block_i, 0, blocktype="down")
        # 3. wrap midblock forward
        self.wrap_block_forward(
            self.unet.mid_block,
            "mid_block",
            0,
            0,
            blocktype="mid")
        # 4. wrap upblock forward
        block_num = len(self.unet.up_blocks)
        for block_i, block in enumerate(self.unet.up_blocks):
            self.wrap_block_forward(
                block, "block", block_num - block_i - 1, 0, blocktype="up"
            )

    def unwrap_modules(self):
        # 1. unet forward
        self.unet.forward = self.function_dict["unet_forward"]
        # 2. downblock forward
        for block_i, block in enumerate(self.unet.down_blocks):
            if ("down", "block", block_i, 0) in self.ignore_block_names:
                continue
            block.forward = self.function_dict[("down", "block", block_i, 0)]
        # 3. midblock forward
        if ("mid", "mid_block", 0, 0) not in self.ignore_block_names:
            self.unet.mid_block.forward = self.function_dict[(
                "mid", "mid_block", 0, 0)]
        # 4. upblock forward
        block_num = len(self.unet.up_blocks)
        for block_i, block in enumerate(self.unet.up_blocks):
            if ("up", "block", block_num - block_i - 1,
                    0) in self.ignore_block_names:
                continue
            block.forward = self.function_dict[
                ("up", "block", block_num - block_i - 1, 0)
            ]

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None

    def frozen_pipe(self):
        for name, attr in vars(self.unet).items():
            if isinstance(attr, torch.nn.Parameter) or isinstance(
                attr, torch.nn.Module
            ):
                print("Freezing:", name)
                attr.requires_grad_(False)

    def inject_cache_mask_module(self):
        num_timesteps = len(list(self.timesteps))
        block_names = self.block_names
        num_blocks = len(block_names)

        all_step_cache_masks = []
        self.query_embed = nn.Embedding(
            num_timesteps * num_blocks, self.hidden_dim)

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
                mask_for_one_time_step = []
                for _ in block_names:
                    mlp = MLP(
                        self.hidden_dim,
                        self.hidden_dim // 4,
                        2,
                        num_layers=self.mlp_layer_num,
                        act=self.activation_type,
                    )
                    mask_for_one_time_step.append(nn.ModuleDict({"mlp": mlp}))
                all_step_cache_masks.append(
                    nn.ModuleList(mask_for_one_time_step))
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
                mask_for_one_time_step = _get_clones(head_mlp, num_blocks)

                all_step_cache_masks.append(
                    nn.ModuleList([pre_mlp, mask_for_one_time_step])
                )
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
                "timesteps": self.timesteps,
            }
        else:
            state_dict = {
                "query_embed": self.query_embed.state_dict(),
                "cache_mask": self.cache_mask.state_dict(),
                "timesteps": self.timesteps,
            }
        torch.save(state_dict, path)

    def load_state_dict(self, state_dict, strict: bool = True):
        print(state_dict.keys())
        print(state_dict["query_embed"].keys())
        print(self.query_embed.state_dict()["weight"].shape)
        try:
            self.query_embed.load_state_dict(state_dict["query_embed"])
        except BaseException:
            weight = torch.zeros_like(self.query_embed.weight.data)
            weight = weight.reshape(
                len(list(self.timesteps)), -1, weight.shape[-1])
            for i in range(len(list(self.timesteps))):
                weight[i] = state_dict["query_embed"][f"{i}.weight"]
            self.query_embed.weight.data = weight.reshape(-1, weight.shape[-1])

        self.cache_mask.load_state_dict(state_dict["cache_mask"])

        if "encoder" in state_dict.keys() and self.encoder is not None:
            self.encoder.load_state_dict(state_dict["encoder"])
        else:
            self.encoder = None
            print("Missing Key: encoder.")
        self.timesteps = state_dict["timesteps"]

    def all_queries_forward(self, update=True):
        # forward queries of all time steps in a once
        all_timestep_gates = []
        all_timestep_original_gates = []
        all_timestep_logits = []
        all_timestep_scores = []

        query_embed = self.query_embed.weight
        query_embed = query_embed.reshape(
            len(list(self.timesteps)), len(self.block_names), self.hidden_dim
        )

        if self.encoder is not None and self.use_attn:
            query_embed = self.encoder(query_embed)

        for i in range(len(self.timesteps)):
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
                [self.block_flops_ratios[x] for x in cascade_nodes[block_name]]
            )
            cascade_flops[block_name] = sum(
                [self.block_flops[x] for x in cascade_nodes[block_name]]
            )
        return cascade_flops, cascade_ratios

    @torch.no_grad()
    def prune(self, expected_ratio, prune_threshold=0.95):
        # expected_ratio: ratios for pruned flops per time steps
        original_block_flops, original_block_ratios = (
            self.block_flops,
            self.block_flops_ratios,
        )
        cascade_nodes = self.get_cascade_nodes()
        device = self.unet.device

        # step1: set target block (from up to down) and get expected flops
        # step2: rank target block probs from high to low, and prune with the prune_threshold and not isreuse already and if less than the ratio
        # step3: update the gate
        # step4: set target block = next(target block) and jump to step2

        t_num = len(self.timesteps)

        all_timestep_gates = [
            {
                block_name: torch.Tensor([0, 1]).to(device)
                for block_name in self.block_names
            }
            for _ in range(t_num)
        ]
        all_timestep_scores = self.all_queries_forward(update=False)[3]

        actual_pruned_ratio = 0.0
        while actual_pruned_ratio < expected_ratio:
            for i, block_name in enumerate(self.block_names[::-1]):
                all_timestep_target_scores = []
                all_timestep_target_gates = []
                for t in range(t_num):
                    target_scores = all_timestep_scores[t][block_name]
                    target_gates = all_timestep_gates[t][block_name]
                    all_timestep_target_scores.append(target_scores)
                    all_timestep_target_gates.append(target_gates)

                sorted_indices = sorted(
                    range(len(all_timestep_target_scores)),
                    key=lambda k: all_timestep_target_scores[k][0],
                    reverse=True,
                )
                filtered_indices = list(
                    filter(
                        lambda x: not self.is_reuse(all_timestep_gates[x][block_name])
                        and all_timestep_target_scores[x][0] > prune_threshold
                        and x > 0,
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
                    for cascade_block_name in cascade_nodes[block_name]:
                        if (
                            all_timestep_gates[pruned_t][cascade_block_name]
                            == torch.Tensor([0, 1]).to(device)
                        ).all():
                            cur_time_pruned_ratios += original_block_ratios[
                                cascade_block_name
                            ]

                    if (
                        actual_pruned_ratio
                        + (cur_pruned_ratios + cur_time_pruned_ratios)
                        / len(self.timesteps)
                    ) >= expected_ratio:
                        break

                    # 2. pruned the specified nodes (set gates to reuse)
                    for cascade_block_name in cascade_nodes[block_name]:
                        if (
                            all_timestep_gates[pruned_t][cascade_block_name]
                            == torch.Tensor([0, 1]).to(device)
                        ).all():
                            all_timestep_gates[pruned_t][cascade_block_name] = (
                                torch.Tensor([1, 0]).to(device))

                    # 3. count the pruned ratios
                    cur_pruned_ratios += cur_time_pruned_ratios

                cur_pruned_ratios = cur_pruned_ratios / len(self.timesteps)
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
