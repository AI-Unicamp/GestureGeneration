import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import betabinom

class Linear(nn.Module):
    """Linear layer with a specific initialization.
    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the layer. Defaults to True.
        init_gain (str, optional): method to compute the gain in the weight initializtion based on the nonlinear activation used afterwards. Defaults to 'linear'.
    """

    def __init__(self, in_features, out_features, bias=True, init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LinearBN(nn.Module):
    """Linear layer with Batch Normalization.
    x -> linear -> BN -> o
    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int ): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the linear layer. Defaults to True.
        init_gain (str, optional): method to set the gain for weight initialization. Defaults to 'linear'.
    """

    def __init__(self, in_features, out_features, bias=True, init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self.batch_normalization = nn.BatchNorm1d(out_features, momentum=0.1, eps=1e-5)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        """
        Shapes:
            x: [T, B, C] or [B, C]
        """
        out = self.linear_layer(x)
        if len(out.shape) == 3:
            out = out.permute(1, 2, 0)
        out = self.batch_normalization(out)
        if len(out.shape) == 3:
            out = out.permute(2, 0, 1)
        return out


class Prenet(nn.Module):
    """Tacotron specific Prenet with an optional Batch Normalization.
    Note:
        Prenet with BN improves the model performance significantly especially
    if it is enabled after learning a diagonal attention alignment with the original
    prenet. However, if the target dataset is high quality then it also works from
    the start. It is also suggested to disable dropout if BN is in use.
        prenet_type == "original"
            x -> [linear -> ReLU -> Dropout]xN -> o
        prenet_type == "bn"
            x -> [linear -> BN -> ReLU -> Dropout]xN -> o
    Args:
        in_features (int): number of channels in the input tensor and the inner layers.
        prenet_type (str, optional): prenet type "original" or "bn". Defaults to "original".
        prenet_dropout (bool, optional): dropout rate. Defaults to True.
        dropout_at_inference (bool, optional): use dropout at inference. It leads to a better quality for some models.
        out_features (list, optional): List of output channels for each prenet block.
            It also defines number of the prenet blocks based on the length of argument list.
            Defaults to [256, 256].
        bias (bool, optional): enable/disable bias in prenet linear layers. Defaults to True.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        in_features,
        prenet_type="original",
        prenet_dropout=True,
        dropout_at_inference=False,
        out_features=[256, 256],
        bias=True,
    ):
        super().__init__()
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        self.dropout_at_inference = dropout_at_inference
        in_features = [in_features] + out_features[:-1]
        if prenet_type == "bn":
            self.linear_layers = nn.ModuleList(
                [LinearBN(in_size, out_size, bias=bias) for (in_size, out_size) in zip(in_features, out_features)]
            )
        elif prenet_type == "original":
            self.linear_layers = nn.ModuleList(
                [Linear(in_size, out_size, bias=bias) for (in_size, out_size) in zip(in_features, out_features)]
            )

    def forward(self, x):
        for linear in self.linear_layers:
            if self.prenet_dropout:
                x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training or self.dropout_at_inference)
            else:
                x = F.relu(linear(x))
        return x

class LocationLayer(nn.Module):
    """Layers for Location Sensitive Attention
    Args:
        attention_dim (int): number of channels in the input tensor.
        attention_n_filters (int, optional): number of filters in convolution. Defaults to 32.
        attention_kernel_size (int, optional): kernel size of convolution filter. Defaults to 31.
    """

    def __init__(self, attention_dim, attention_n_filters=32, attention_kernel_size=31):
        super().__init__()
        self.location_conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False,
        )
        self.location_dense = Linear(attention_n_filters, attention_dim, bias=False, init_gain="tanh")

    def forward(self, attention_cat):
        """
        Shapes:
            attention_cat: [B, 2, C]
        """
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention


class GravesAttention(nn.Module):
    """Graves Attention as is ref1 with updates from ref2.
    ref1: https://arxiv.org/abs/1910.10288
    ref2: https://arxiv.org/pdf/1906.01083.pdf
    Args:
        query_dim (int): number of channels in query tensor.
        K (int): number of Gaussian heads to be used for computing attention.
    """

    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):

        super().__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True), nn.ReLU(), nn.Linear(query_dim, 3 * K, bias=True)
        )
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2 * self.K) : (3 * self.K)], 1.0)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K : (2 * self.K)], 10)  # bias std

    def init_states(self, inputs):
        if self.J is None or inputs.shape[1] + 1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1] + 2.0).to(inputs.device) + 0.5
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs, processed_inputs, mask):
        """
        Shapes:
            query: [B, C_attention_rnn]
            inputs: [B, T_in, C_encoder]
            processed_inputs: place_holder
            mask: [B, T_in]
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # dropout to decorrelate attention heads
        g_t = torch.nn.functional.dropout(g_t, p=0.5, training=self.training)

        # attention GMM parameters
        sig_t = torch.nn.functional.softplus(b_t) + self.eps

        mu_t = self.mu_prev + torch.nn.functional.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[: inputs.size(1) + 1]

        # attention weights
        phi_t = g_t.unsqueeze(-1) * (1 / (1 + torch.sigmoid((mu_t.unsqueeze(-1) - j) / sig_t.unsqueeze(-1))))

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = 1e-8

        # apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)
        self.attention_weights = alpha_t
        self.mu_prev = mu_t
        return context


class OriginalAttention(nn.Module):
    """Bahdanau Attention with various optional modifications.
    - Location sensitive attnetion: https://arxiv.org/abs/1712.05884
    - Forward Attention: https://arxiv.org/abs/1807.06736 + state masking at inference
    - Using sigmoid instead of softmax normalization
    - Attention windowing at inference time
    Note:
        Location Sensitive Attention extends the additive attention mechanism
    to use cumulative attention weights from previous decoder time steps with the current time step features.
        Forward attention computes most probable monotonic alignment. The modified attention probabilities at each
    timestep are computed recursively by the forward algorithm.
        Transition agent in the forward attention explicitly gates the attention mechanism whether to move forward or
    stay at each decoder timestep.
        Attention windowing is a inductive prior that prevents the model from attending to previous and future timesteps
    beyond a certain window.
    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the vakue tensor. In general, the value tensor is the output of the encoder layer.
        attention_dim (int): number of channels of the inner attention layers.
        location_attention (bool): enable/disable location sensitive attention.
        attention_location_n_filters (int): number of location attention filters.
        attention_location_kernel_size (int): filter size of location attention convolution layer.
        windowing (int): window size for attention windowing. if it is 5, for computing the attention, it only considers the time steps [(t-5), ..., (t+5)] of the input.
        norm (str): normalization method applied to the attention weights. 'softmax' or 'sigmoid'
        forward_attn (bool): enable/disable forward attention.
        trans_agent (bool): enable/disable transition agent in the forward attention.
        forward_attn_mask (int): enable/disable an explicit masking in forward attention. It is useful to set at especially inference time.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        query_dim,
        embedding_dim,
        attention_dim,
        location_attention,
        attention_location_n_filters,
        attention_location_kernel_size,
        windowing,
        norm,
        forward_attn,
        trans_agent,
        forward_attn_mask,
    ):
        super().__init__()
        self.query_layer = Linear(query_dim, attention_dim, bias=False, init_gain="tanh")
        self.inputs_layer = Linear(embedding_dim, attention_dim, bias=False, init_gain="tanh")
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(query_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
            )
        self._mask_value = -float("inf")
        self.windowing = windowing
        self.win_idx = None
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attention = location_attention

    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6

    def init_forward_attn(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat([torch.ones([B, 1]), torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights_cum = torch.zeros([B, T], device=inputs.device)

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        if self.location_attention:
            self.init_location_attention(inputs)
        if self.forward_attn:
            self.init_forward_attn(inputs)
        if self.windowing:
            self.init_win_idx()

    def preprocess_inputs(self, inputs):
        return self.inputs_layer(inputs)

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_attention(self, query, processed_inputs):
        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def apply_windowing(self, attention, inputs):
        back_win = self.win_idx - self.win_back
        front_win = self.win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")
        # this is a trick to solve a special problem.
        # but it does not hurt.
        if self.win_idx == -1:
            attention[:, 0] = attention.max()
        # Update the window
        self.win_idx = torch.argmax(attention, 1).long()[0].item()
        return attention

    def apply_forward_attention(self, alignment):
        # forward attention
        fwd_shifted_alpha = F.pad(self.alpha[:, :-1].clone().to(alignment.device), (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha + self.u * fwd_shifted_alpha + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, _ = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3 :] = 0
                alpha[b, : (n[b] - 1)] = 0  # ignore all previous states to prevent repetition.
                alpha[b, (n[b] - 2)] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, processed_inputs, mask):
        """
        shapes:
            query: [B, C_attn_rnn]
            inputs: [B, T_en, D_en]
            processed_inputs: [B, T_en, D_attn]
            mask: [B, T_en]
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(query, processed_inputs)
        else:
            attention, _ = self.get_attention(query, processed_inputs)
        # apply masking
        if mask is not None:
            attention.data.masked_fill_(~mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)

        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(attention).sum(dim=1, keepdim=True)
        else:
            raise ValueError("Unknown value for attention norm type")

        if self.location_attention:
            self.update_location_attention(alignment)

        # apply forward attention if enabled
        if self.forward_attn:
            alignment = self.apply_forward_attention(alignment)
            self.alpha = alignment

        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = alignment

        # compute transition agent
        if self.forward_attn and self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context


class MonotonicDynamicConvolutionAttention(nn.Module):
    """Dynamic convolution attention from
    https://arxiv.org/pdf/1910.10288.pdf
    query -> linear -> tanh -> linear ->|
                                        |                                            mask values
                                        v                                              |    |
               atten_w(t-1) -|-> conv1d_dynamic -> linear -|-> tanh -> + -> softmax -> * -> * -> context
                             |-> conv1d_static  -> linear -|           |
                             |-> conv1d_prior   -> log ----------------|
    query: attention rnn output.
    Note:
        Dynamic convolution attention is an alternation of the location senstive attention with
    dynamically computed convolution filters from the previous attention scores and a set of
    constraints to keep the attention alignment diagonal.
        DCA is sensitive to mixed precision training and might cause instable training.
    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
        static_filter_dim (int): number of channels in the convolution layer computing the static filters.
        static_kernel_size (int): kernel size for the convolution layer computing the static filters.
        dynamic_filter_dim (int): number of channels in the convolution layer computing the dynamic filters.
        dynamic_kernel_size (int): kernel size for the convolution layer computing the dynamic filters.
        prior_filter_len (int, optional): [description]. Defaults to 11 from the paper.
        alpha (float, optional): [description]. Defaults to 0.1 from the paper.
        beta (float, optional): [description]. Defaults to 0.9 from the paper.
    """

    def __init__(
        self,
        query_dim,
        embedding_dim,  # pylint: disable=unused-argument
        attention_dim,
        static_filter_dim,
        static_kernel_size,
        dynamic_filter_dim,
        dynamic_kernel_size,
        prior_filter_len=11,
        alpha=0.1,
        beta=0.9,
    ):
        super().__init__()
        self._mask_value = 1e-8
        self.dynamic_filter_dim = dynamic_filter_dim
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_filter_len = prior_filter_len
        self.attention_weights = None
        # setup key and query layers
        self.query_layer = nn.Linear(query_dim, attention_dim)
        self.key_layer = nn.Linear(attention_dim, dynamic_filter_dim * dynamic_kernel_size, bias=False)
        self.static_filter_conv = nn.Conv1d(
            1,
            static_filter_dim,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.static_filter_layer = nn.Linear(static_filter_dim, attention_dim, bias=False)
        self.dynamic_filter_layer = nn.Linear(dynamic_filter_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        prior = betabinom.pmf(range(prior_filter_len), prior_filter_len - 1, alpha, beta)
        self.register_buffer("prior", torch.FloatTensor(prior).flip(0))

    # pylint: disable=unused-argument
    def forward(self, query, inputs, processed_inputs, mask):
        """
        query: [B, C_attn_rnn]
        inputs: [B, T_en, D_en]
        processed_inputs: place holder.
        mask: [B, T_en]
        """
        # compute prior filters
        prior_filter = F.conv1d(
            F.pad(self.attention_weights.unsqueeze(1), (self.prior_filter_len - 1, 0)), self.prior.view(1, 1, -1)
        )
        prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)
        G = self.key_layer(torch.tanh(self.query_layer(query)))
        # compute dynamic filters
        dynamic_filter = F.conv1d(
            self.attention_weights.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        dynamic_filter = dynamic_filter.view(query.size(0), self.dynamic_filter_dim, -1).transpose(1, 2)
        # compute static filters
        static_filter = self.static_filter_conv(self.attention_weights.unsqueeze(1)).transpose(1, 2)
        alignment = (
            self.v(
                torch.tanh(self.static_filter_layer(static_filter) + self.dynamic_filter_layer(dynamic_filter))
            ).squeeze(-1)
            + prior_filter
        )
        # compute attention weights
        attention_weights = F.softmax(alignment, dim=-1)
        # apply masking
        if mask is not None:
            attention_weights.data.masked_fill_(~mask, self._mask_value)
        self.attention_weights = attention_weights
        # compute context
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def preprocess_inputs(self, inputs):  # pylint: disable=no-self-use
        return None

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        self.attention_weights[:, 0] = 1.0


def init_attn(
    attn_type,
    query_dim,
    embedding_dim,
    attention_dim,
    location_attention,
    attention_location_n_filters,
    attention_location_kernel_size,
    windowing,
    norm,
    forward_attn,
    trans_agent,
    forward_attn_mask,
    attn_K,
):
    if attn_type == "original":
        return OriginalAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            location_attention,
            attention_location_n_filters,
            attention_location_kernel_size,
            windowing,
            norm,
            forward_attn,
            trans_agent,
            forward_attn_mask,
        )
    if attn_type == "graves":
        return GravesAttention(query_dim, attn_K)
    if attn_type == "dynamic_convolution":
        return MonotonicDynamicConvolutionAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            static_filter_dim=8,
            static_kernel_size=21,
            dynamic_filter_dim=8,
            dynamic_kernel_size=21,
            prior_filter_len=11,
            alpha=0.1,
            beta=0.9,
        )

    raise RuntimeError(" [!] Given Attention Type '{attn_type}' is not exist.")

# NOTE: linter has a problem with the current TF release
# pylint: disable=no-value-for-parameter
# pylint: disable=unexpected-keyword-arg
class ConvBNBlock(nn.Module):
    r"""Convolutions with Batch Normalization and non-linear activation.
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): convolution kernel size.
        activation (str): 'relu', 'tanh', None (linear).
    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.convolution1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_normalization = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        o = self.convolution1d(x)
        o = self.batch_normalization(o)
        o = self.activation(o)
        o = self.dropout(o)
        return o


class Postnet(nn.Module):
    r"""Tacotron2 Postnet
    Args:
        in_out_channels (int): number of output channels.
    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, in_out_channels, num_convs=5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(ConvBNBlock(in_out_channels, 512, kernel_size=5, activation="tanh"))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(ConvBNBlock(512, 512, kernel_size=5, activation="tanh"))
        self.convolutions.append(ConvBNBlock(512, in_out_channels, kernel_size=5, activation=None))

    def forward(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        return o


class Encoder(nn.Module):
    r"""Tacotron2 Encoder
    Args:
        in_out_channels (int): number of input and output channels.
    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, in_out_channels=512):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for _ in range(3):
            self.convolutions.append(ConvBNBlock(in_out_channels, in_out_channels, 5, "relu"))
        self.lstm = nn.LSTM(
            in_out_channels, int(in_out_channels / 2), num_layers=1, batch_first=True, bias=True, bidirectional=True
        )
        self.rnn_state = None

    def forward(self, x, input_lengths):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o

    def inference(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        # self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        return o


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    """Tacotron2 decoder. We don't use Zoneout but Dropout between RNN layers.
    Args:
        in_channels (int): number of input channels.
        frame_channels (int): number of feature frame channels.
        r (int): number of outputs per time step (reduction rate).
        memory_size (int): size of the past window. if <= 0 memory_size = r
        attn_type (string): type of attention used in decoder.
        attn_win (bool): if true, define an attention window centered to maximum
            attention response. It provides more robust attention alignment especially
            at interence time.
        attn_norm (string): attention normalization function. 'sigmoid' or 'softmax'.
        prenet_type (string): 'original' or 'bn'.
        prenet_dropout (float): prenet dropout rate.
        forward_attn (bool): if true, use forward attention method. https://arxiv.org/abs/1807.06736
        trans_agent (bool): if true, use transition agent. https://arxiv.org/abs/1807.06736
        forward_attn_mask (bool): if true, mask attention values smaller than a threshold.
        location_attn (bool): if true, use location sensitive attention.
        attn_K (int): number of attention heads for GravesAttention.
        separate_stopnet (bool): if true, detach stopnet input to prevent gradient flow.
        max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 10000.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        in_channels,
        frame_channels,
        r,
        attn_type,
        attn_win,
        attn_norm,
        prenet_type,
        prenet_dropout,
        forward_attn,
        trans_agent,
        forward_attn_mask,
        location_attn,
        attn_K,
        separate_stopnet,
        max_decoder_steps,
    ):
        super().__init__()
        self.frame_channels = frame_channels
        self.r_init = r
        self.r = r
        self.encoder_embedding_dim = in_channels
        self.separate_stopnet = separate_stopnet
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = 0.5

        # model dimensions
        self.query_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.attn_dim = 128
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # memory -> |Prenet| -> processed_memory
        prenet_dim = self.frame_channels
        self.prenet = Prenet(
            prenet_dim, prenet_type, prenet_dropout, out_features=[self.prenet_dim, self.prenet_dim], bias=False
        )

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_channels, self.query_dim, bias=True)

        self.attention = init_attn(
            attn_type=attn_type,
            query_dim=self.query_dim,
            embedding_dim=in_channels,
            attention_dim=128,
            location_attention=location_attn,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            windowing=attn_win,
            norm=attn_norm,
            forward_attn=forward_attn,
            trans_agent=trans_agent,
            forward_attn_mask=forward_attn_mask,
            attn_K=attn_K,
        )

        self.decoder_rnn = nn.LSTMCell(self.query_dim + in_channels, self.decoder_rnn_dim, bias=True)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_channels, self.frame_channels * self.r_init)

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(self.decoder_rnn_dim + self.frame_channels * self.r_init, 1, bias=True, init_gain="sigmoid"),
        )
        self.memory_truncated = None

    def set_r(self, new_r):
        self.r = new_r

    def get_go_frame(self, inputs):
        B = inputs.size(0)
        memory = torch.zeros(1, device=inputs.device).repeat(B, self.frame_channels * self.r)
        return memory

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)
        # T = inputs.size(1)
        if not keep_states:
            self.query = torch.zeros(1, device=inputs.device).repeat(B, self.query_dim)
            self.attention_rnn_cell_state = torch.zeros(1, device=inputs.device).repeat(B, self.query_dim)
            self.decoder_hidden = torch.zeros(1, device=inputs.device).repeat(B, self.decoder_rnn_dim)
            self.decoder_cell = torch.zeros(1, device=inputs.device).repeat(B, self.decoder_rnn_dim)
            self.context = torch.zeros(1, device=inputs.device).repeat(B, self.encoder_embedding_dim)
        self.inputs = inputs
        self.processed_inputs = self.attention.preprocess_inputs(inputs)
        self.mask = mask

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.frame_channels:
            memory = memory.view(memory.shape[0], memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, frame_channels)
        memory = memory.transpose(0, 1)
        return memory

    def _parse_outputs(self, outputs, stop_tokens, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.frame_channels)
        outputs = outputs.transpose(1, 2)
        return outputs, stop_tokens, alignments

    def _update_memory(self, memory):
        if len(memory.shape) == 2:
            return memory[:, self.frame_channels * (self.r - 1) :]
        return memory[:, :, self.frame_channels * (self.r - 1) :]

    def decode(self, memory):
        """
        shapes:
           - memory: B x r * self.frame_channels
        """
        # self.context: B x D_en
        # query_input: B x D_en + (r * self.frame_channels)
        query_input = torch.cat((memory, self.context), -1)
        # self.query and self.attention_rnn_cell_state : B x D_attn_rnn
        self.query, self.attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state)
        )
        self.query = F.dropout(self.query, self.p_attention_dropout, self.training)
        self.attention_rnn_cell_state = F.dropout(
            self.attention_rnn_cell_state, self.p_attention_dropout, self.training
        )
        # B x D_en
        self.context = self.attention(self.query, self.inputs, self.processed_inputs, self.mask)
        # B x (D_en + D_attn_rnn)
        decoder_rnn_input = torch.cat((self.query, self.context), -1)
        # self.decoder_hidden and self.decoder_cell: B x D_decoder_rnn
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        # B x (D_decoder_rnn + D_en)
        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context), dim=1)
        # B x (self.r * self.frame_channels)
        decoder_output = self.linear_projection(decoder_hidden_context)
        # B x (D_decoder_rnn + (self.r * self.frame_channels))
        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)
        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        # select outputs for the reduction rate self.r
        decoder_output = decoder_output[:, : self.r * self.frame_channels]
        return decoder_output, self.attention.attention_weights, stop_token

    def forward(self, inputs, memories, mask):
        r"""Train Decoder with teacher forcing.
        Args:
            inputs: Encoder outputs.
            memories: Feature frames for teacher-forcing.
            mask: Attention mask for sequence padding.
        Shapes:
            - inputs: (B, T, D_out_enc)
            - memory: (B, T_mel, D_mel)
            - outputs: (B, T_mel, D_mel)
            - alignments: (B, T_in, T_out)
            - stop_tokens: (B, T_out)
        """
        memory = self.get_go_frame(inputs).unsqueeze(0)
        memories = self._reshape_memory(memories)
        memories = torch.cat((memory, memories), dim=0)
        memories = self._update_memory(memories)
        memories = self.prenet(memories)

        self._init_states(inputs, mask=mask)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            decoder_output, attention_weights, stop_token = self.decode(memory)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [attention_weights]

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)
        return outputs, alignments, stop_tokens

    def inference(self, inputs):
        r"""Decoder inference without teacher forcing and use
        Stopnet to stop decoder.
        Args:
            inputs: Encoder outputs.
        Shapes:
            - inputs: (B, T, D_out_enc)
            - outputs: (B, T_mel, D_mel)
            - alignments: (B, T_in, T_out)
            - stop_tokens: (B, T_out)
        """
        memory = self.get_go_frame(inputs)
        memory = self._update_memory(memory)

        self._init_states(inputs, mask=None)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments, t = [], [], [], 0
        while True:
            memory = self.prenet(memory)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > self.stop_threshold and t > inputs.shape[0] // 2:
                break
            if len(outputs) == self.max_decoder_steps:
                print(f"   > Decoder stopped with `max_decoder_steps` {self.max_decoder_steps}")
                break

            memory = self._update_memory(decoder_output)
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_truncated(self, inputs):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None, keep_states=False)
        else:
            self._init_states(inputs, mask=None, keep_states=True)

        self.attention.init_states(inputs)
        outputs, stop_tokens, alignments, t = [], [], [], 0
        while True:
            memory = self.prenet(self.memory_truncated)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > 0.7:
                break
            if len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

            self.memory_truncated = decoder_output
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_step(self, inputs, t, memory=None):
        """
        For debug purposes
        """
        if t == 0:
            memory = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None)

        memory = self.prenet(memory)
        decoder_output, stop_token, alignment = self.decode(memory)
        stop_token = torch.sigmoid(stop_token.data)
        memory = decoder_output
        return decoder_output, stop_token, alignment

class GST(nn.Module):
    """Global Style Token Module for factorizing prosody in speech.
    See https://arxiv.org/pdf/1803.09017"""

    def __init__(self, num_mel, num_heads, num_style_tokens, gst_embedding_dim, embedded_speaker_dim=None):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mel, gst_embedding_dim)
        self.style_token_layer = StyleTokenLayer(num_heads, num_style_tokens, gst_embedding_dim, embedded_speaker_dim)

    def forward(self, inputs, speaker_embedding=None):
        enc_out = self.encoder(inputs)
        # concat speaker_embedding
        if speaker_embedding is not None:
            enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)
        style_embed = self.style_token_layer(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.
    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]

    Modified version to deal with mocap sequence modeling

    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim // 2, batch_first=True
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        # _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        out, _ = self.recurrence(x)
        # out: 3D tensor [batch_size, seq_len, encoding_dize]

        return out

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


class StyleTokenLayer(nn.Module):
    """NN Module attending to style tokens based on prosody encodings."""

    def __init__(self, num_heads, num_style_tokens, gst_embedding_dim, d_vector_dim=None):
        super().__init__()

        self.query_dim = gst_embedding_dim // 2

        if d_vector_dim:
            self.query_dim += d_vector_dim

        self.key_dim = gst_embedding_dim // num_heads
        self.style_tokens = nn.Parameter(torch.FloatTensor(num_style_tokens, self.key_dim))
        nn.init.normal_(self.style_tokens, mean=0, std=0.5)
        self.attention = MultiHeadAttention(
            query_dim=self.query_dim, key_dim=self.key_dim, num_units=gst_embedding_dim, num_heads=num_heads
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        prosody_encoding = inputs.unsqueeze(1)
        # prosody_encoding: 3D tensor [batch_size, 1, encoding_size==128]
        tokens = torch.tanh(self.style_tokens).unsqueeze(0).expand(batch_size, -1, -1)
        # tokens: 3D tensor [batch_size, num tokens, token embedding size]
        style_embed = self.attention(prosody_encoding, tokens)

        return style_embed


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        queries = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        queries = torch.stack(torch.split(queries, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(queries, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
