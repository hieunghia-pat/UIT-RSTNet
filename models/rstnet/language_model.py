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
    def __init__(self, vocab, padding_idx=0, bert_hidden_size=768, vocab_size=10201, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, max_len=54, dropout=.1):
        super(LanguageModel, self).__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model

        self.language_model = RobertaModel.from_pretrained("vinai/phobert-base", return_dict=True)
        self.language_model.config.vocab_size = vocab_size
        self.proj_to_caption_model = nn.Linear(bert_hidden_size, d_model)

        self.vocab = vocab
        self.pretrained_language_idx_mapping = vocab.pretrained_language_idx_mapping
        self.pretrained_padding_idx = vocab.pretrained_language_idx_mapping[self.vocab.padding_idx]

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.encoder_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.proj_to_vocab = nn.Linear(d_model, vocab_size)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=False,
        output_hidden_states=False, return_dict=False, encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        pretrained_input_ids = []
        
        for input_id in input_ids:
            pretrained_input_id = [self.pretrained_language_idx_mapping[str(ori_id.item())] for ori_id in input_id]
            pretrained_input_ids.append(pretrained_input_id)

        # List -> tensor.
        pretrained_input_ids = torch.tensor(pretrained_input_ids).to(input_ids.device)

        # input (b_s, seq_len)
        b_s, seq_len = pretrained_input_ids.shape[:2]
        mask_queries = (pretrained_input_ids != self.pretrained_padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=pretrained_input_ids.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (pretrained_input_ids == self.pretrained_padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(pretrained_input_ids.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if attention_mask is None:
            attention_mask = torch.ones_like(pretrained_input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(pretrained_input_ids).long()

        bert_output = self.language_model(
            input_ids=pretrained_input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        language_feature = self.proj_to_caption_model(bert_output.last_hidden_state)
        language_feature = language_feature + self.pos_emb(seq)

        language_feature = self.encoder_layer(language_feature, mask_queries, mask_self_attention)

        logits = self.proj_to_vocab(language_feature)
        out = F.log_softmax(logits, dim=-1)
        return out, language_feature
