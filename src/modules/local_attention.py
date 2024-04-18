# copied from https://github.com/lucidrains/local-attention/blob/master/local_attention/transformer.py

from local_attention.transformer import *


class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        dim,
        depth,
        positional_embeding=True,
        causal=True,
        local_attn_window_size=512,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ignore_index=-1,
        use_xpos=False,
        xpos_scale_base=None,
        use_dynamic_pos_bias=False,
        **kwargs
    ):
        super().__init__()

        self.positional_embeding = positional_embeding
        if positional_embeding:
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=causal,
                            window_size=local_attn_window_size,
                            use_xpos=use_xpos,
                            xpos_scale_base=xpos_scale_base,
                            use_rotary_pos_emb=not use_dynamic_pos_bias,
                            prenorm=True,
                            **kwargs
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.ignore_index = ignore_index

    def forward(self, hidden_states, mask=None):

        n, device = hidden_states.shape[1], hidden_states.device

        assert n <= self.max_seq_len
        if self.positional_embeding:
            hidden_states = hidden_states + self.pos_emb(torch.arange(n, device=device))

        # dynamic pos bias

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        for attn, ff in self.layers:
            hidden_states = (
                attn(hidden_states, mask=mask, attn_bias=attn_bias) + hidden_states
            )
            hidden_states = ff(hidden_states) + hidden_states

        return hidden_states


class local_attention_block(nn.Module):
    def __init__(
        self,
        input_dims=256,
        depth=2,
        local_attn_window_size=64,
        max_seq_len=1000,
        positional_embeding=False,
    ):
        super(local_attention_block, self).__init__()

        self.input_dims = input_dims
        self.output_dims = input_dims

        self.model = LocalTransformer(
            dim=input_dims,
            depth=depth,
            local_attn_window_size=local_attn_window_size,
            max_seq_len=max_seq_len,
            positional_embeding=positional_embeding,
        )

    def forward(self, hidden_states):

        hidden_states = self.model(hidden_states)

        return hidden_states
