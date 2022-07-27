import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaModel

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module


class EncoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, mask_pad, mask_self_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad

        # FFN+AddNorm
        ff = self.pwff(self_att)
        ff = ff * mask_pad
        return ff


class LanguageModel(Module):
    def __init__(self, padding_idx=0, bert_hidden_size=768, vocab_size=10201, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, max_len=54, dropout=.1):
        super(LanguageModel, self).__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model

        self.proj_to_bert_model = nn.Linear(d_model, bert_hidden_size)
        language_model = RobertaModel.from_pretrained("vinai/phobert-base", return_dict=True)
        self.language_model_encoder = language_model.encoder
        self.language_model_pooler = language_model.pooler
        self.proj_to_caption_model = nn.Linear(bert_hidden_size, d_model)

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.encoder_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.proj_to_vocab = nn.Linear(d_model, vocab_size)

        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, inputs, mask_self_attention, mask_queries):
        # inputs (b_s, seq_len, d_model)
        b_s, seq_len = inputs.shape[:2]
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(inputs.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        inputs = self.proj_to_bert_model(inputs)
        bert_encoder_output = self.language_model_encoder(hidden_states=inputs, 
                                                            attention_mask=mask_queries.squeeze(-1).unsqueeze(1).unsqueeze(1)).last_hidden_state
        linguistic_output = self.language_model_pooler(bert_encoder_output)
        language_feature = self.proj_to_caption_model(linguistic_output)
        language_feature = language_feature + self.pos_emb(seq)

        language_feature = self.encoder_layer(language_feature, mask_queries, mask_self_attention)

        logits = self.proj_to_vocab(language_feature)
        out = F.log_softmax(logits, dim=-1)
        return out, language_feature
