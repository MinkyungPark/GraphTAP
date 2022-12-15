from torch.autograd import Function
from examples.graph_diffusion.models.transformer import *
from examples.graph_diffusion.models.autuencoder import SymbolWiseTransformer

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            codebook = codebook.to(dtype=torch.float32)
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbeddingMovingAverage(nn.Module):
    def __init__(self, K, D, decay=0.99):
        super().__init__()
        embedding = torch.zeros(K, D)
        embedding.uniform_(-1./K, 1./K)
        self.decay = decay

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_w", self.embedding.clone())

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x): # traj feature
        K, D = self.embedding.size()
        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding)
        z_q_x = z_q_x_.contiguous()

        if self.training:
            encodings = F.one_hot(indices, K).float()
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
            self.embedding = self.embedding.detach()
            self.ema_w = self.ema_w.detach()
            self.ema_count = self.ema_count.detach()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar

########
from fairseq.models import FairseqDecoder

class VQTransformerDecoder(FairseqDecoder):
    def __init__(self, config):
        super().__init__(dictionary=None)
        
        # self.K=config.K
        self.K=config.K
        self.latent_size = (config.num_node + 1) * config.enc_embed_dim
        self.embedding_dim = config.n_embd
        
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        
        self.state_conditional = config.state_conditional

        # codebook
        self.codebook = VQEmbeddingMovingAverage(self.latent_size, self.latent_size)
        self.ma_update = True

        self.residual = config.residual
        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        sub_sample_size = config.block_size // self.transition_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, sub_sample_size, config.n_embd))
        # self.cast_embed = nn.Linear(self.embedding_dim, self.latent_size)
        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)
        self.latent_mixing = nn.Linear(self.latent_size + self.observation_dim, self.embedding_dim)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, graph_features, init_state):
        """
            latents: [B x (T//self.latent_step) x latent_size]
            state: [B x observation_dimension]
        """
        latents_st, latents = self.codebook.straight_through(graph_features) # B, T, N * F_D
        
        B, T, _ = latents_st.shape
        state_flat = torch.reshape(init_state, shape=[B, 1, -1]).repeat(1, T, 1)
        
        if not self.state_conditional:
            state_flat = torch.zeros_like(state_flat)
        
        inputs = torch.cat([state_flat, latents_st], dim=-1)
        inputs = self.latent_mixing(inputs)

        inputs = inputs + self.pos_emb[:, :inputs.shape[1]]
        x = self.decoder(inputs)
        x = self.ln_f(x)

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        joined_pred[:, :, :self.observation_dim] += torch.reshape(init_state, shape=[B, 1, -1])
        return joined_pred, latents, graph_features


