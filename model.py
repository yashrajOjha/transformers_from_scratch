import torch
import math
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # multiplied by the square root of the embedding dimension to rescale with positional embedding values
        # increase the embedding values before the addition is to make the positional encoding relatively smaller
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of seq_len * d_model
        pe = torch.zeroes(seq_len, d_model)

        # PE (pos, 2i) = sin (pos/10000 ^ 2i/d_model), even
        # PE (pos, 2i + 1) = cos (pos/10000 ^ 2i/d_model), odd

        #vector representing position inside sequence
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # 1D tensor into a 2D tensor

        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/ d_model))

        pe[:, 0::2] = torch.sin(position*division_term)
        pe[:, 1::2] = torch.cos(position*division_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # not as a learned parameter, but rather as a saved parameter, saved as file

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # requires_grad_ is a method that sets your tensors requires_grad attribute to True
        # exactly the right amount of positional encoding information to match your input's sequence dimension -> x.shape[1]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeroes(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)  #meam cancels the dimensions to which it is applied, we want to keep it
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)  # w2 and b2

    def forward(self, x):
        # batch, seq_len, d_model -> convert using linear_1 to get (batch, seq_len, d_ff) -> linear_2 to convert it back
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by number of heads"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # [batch, head, seq_len, d_k] -> [batch, head, seq_len, seq_len]
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask:
            attention_scores.masked_fill(mask==0, -1e9) #we do not want some interaction between tokens, to not attend to padded
        
        attention_scores = attention_scores.softmax(dim = -1) # [batch, head, seq_len, seq_len]

        if dropout:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # [batch_size, seq_len, d_model]
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # number of sequence, number of tokens in each sequence, number of heads, reduced dimensions
        # we would ideally want, different heads to focus on different parts of the sentence, different relationships between tokens
        # [batch_size, seq_length, num_heads, head_dimension], transpose to [batch_size, num_heads, seq_length, head_dimension]
        # each attention head gets its own "slice" of the representation
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, dropout)

        # transpose swaps dimensions back to [batch_size, seq_len, h, d_k]
        # view collapses the last two dimensions, resulting in shape [batch_size, seq_len, d_model]
        # The -1 in the view automatically calculates the appropriate dimension (seq_len)
        # The .contiguous() call is particularly important because after transposing, the tensor's memory layout may not be contiguous, which is required for the subsequent view operation. 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # [batch, seq_len, d_model] to [batch, seq_len, d_model]
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        # we dont want padding to interact with other words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #calling forward of attention
        x = self.residual_connections[1](x, self.feed_forward_block) #passing input through just the feedforward block
        return x

# many encoder blocks
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, 
    self_attention_block: MultiHeadAttentionBlock, 
    cross_attention_block: MultiHeadAttentionBlock, 
    feed_forward_block: FeedForwardBlock, 
    dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # each token's representation from d_model dimensions to vocab_size dimensions

    def forward(self, x):
        # [Batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
        return torch.log_softmax(self.proj(x), dim = -1) # converts raw logits into log probabilities, softmax to be applied across last dimension

class Transformer(nn.Module):
    def __init__(self, 
    encoder: Encoder, 
    decoder: Decoder, 
    src_embedding: InputEmbedding, 
    target_embedding: InputEmbedding, 
    src_pos: PositionalEncoding, 
    target_pos: PositionalEncoding, 
    projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, 
    target_vocab_size: int, 
    src_seq_len: int, 
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: int = 0.1,
    d_ff: int = 2048
    ):
    # embedding layers
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    target_embedding = InputEmbedding(d_model, target_vocab_size)

    # positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_mode, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention,
            decoder_cross_attention,
            feed_forward_block,
            dropout
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embedding,
        target_embedding, 
        src_pos,
        target_pos, 
        projection_layer
    )

    # initialize parameters, xavier
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer