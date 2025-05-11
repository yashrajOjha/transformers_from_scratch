import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 #everything that is 1 will become false, so lower traingle is available for attention
    
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang

        self.sos_token = torch.Tensor([
            tokenizer_src.token_to_id(['[SOS]'])
        ], dtype= torch.int64)

        self.eos_token = torch.Tensor([
            tokenizer_src.token_to_id(['[EOS]'])
        ], dtype= torch.int64)

        self.pad_token = torch.Tensor([
            tokenizer_src.token_to_id(['[PAD]'])
        ], dtype= torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        # convert to token then to input ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        # add padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # for start and end
        dec_num_paddig_tokens = self.seq_len - len(dec_input_tokens) - 1 # only the start of sentence token

        if enc_num_padding_tokens < 0  or dec_num_paddig_tokens < 0:
            raise ValueError("sentence is too long")

        # one sentence is output of the decoder, is label/target, one tensor for encoder input, one tensor for decoder input
        # adding sos token, actual input tokens, then eos token and remaining padding tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype= torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_paddig_tokens, dtype=  torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype= torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_paddig_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # mask operations are to be broadcastable in the attention mechanism's matrix multiplications
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # [1, 1, seq_len] # not to focus on padding tokens
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int & causal_mask(decoder_input.size(0)), # [1, seq_len] & [1, seq_len, seq_len]
            "label":  label,
            "src_text": src_text,
            "target_text": target_text
        }