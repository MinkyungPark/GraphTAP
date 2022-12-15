from examples.graph_diffusion.models.transformer import *

class TransformerPrior(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.K, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.state_emb = nn.Linear(config.observation_dim, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.K, bias=False)
        # self.head = EinLinear(config.transition_dim, config.n_embd, config.K, bias=False)
        self.observation_dim = config.observation_dim

        self.vocab_size = config.K
        self.block_size = config.block_size
        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, state, targets=None):
        """
            idx : [ B x T ]
            state: [ B ]
        """

        state = state.to(dtype=torch.float32)
        ## [ B x T x embedding_dim ]
        if not idx is None:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            token_embeddings = torch.cat([torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings), token_embeddings],
                                             dim=1)
        else:
            b = 1
            t = 0
            token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state)

        ## [ 1 x T+1 x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t+1, :] # each position maps to a (learnable) vector
        state_embeddings = self.state_emb(state)[:, None]
        ## [ B x T+1 x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings + state_embeddings)
        x = self.blocks(x)
        ## [ B x T+1 x embedding_dim ]
        x = self.ln_f(x)

        logits = self.head(x)
        logits = logits.reshape(b, t + 1, self.vocab_size)
        logits = logits[:,:t+1]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape([-1]), reduction='none')
            loss = loss.mean()
        else:
            loss = None

        return logits, loss
