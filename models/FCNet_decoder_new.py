from config import Constants
import torch
import torch.nn as nn
from .bert import BertEmbeddings, BertLayer
from .self_attention import self_attention_layer
from torch.nn import Parameter
import torch.nn.functional as F

# from models import LSTM_squence, LSTM_squence_v2

__all__ = ('BertDecoder', 'BertDecoderDisentangled')

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sim_non_pad_mask(seq):
    assert seq.dim() == 2
    bs = seq.size(0)
    max_len = seq.size(1)
    before_mask = seq.ne(Constants.PAD)
    for j in range(bs):
        for i in range(max_len):
            if before_mask[j, max_len-i-1] == 1:
                before_mask[j, max_len-i-1] = 0
                continue
    return before_mask.type(torch.float).unsqueeze(-1)


# 对seq_k的非pad进行mask，上面这个函数则是对pad进行mask
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_attn_sim_key_pad_mask(seq_k, seq_q):
    bs = seq_q.size(0)
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)  # bs x lk
    for i in range(bs):
        for j in range(len_q):
            if padding_mask[i, len_q-1-j] == 0:
                padding_mask[i, len_q-1-j] == 1
                continue
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # bs x lq x lk
    return padding_mask

def get_subsequent_mask(seq, watch=0):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    if watch != 0 and len_s >= watch:
        assert watch > 0
        tmp = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=-watch)
    else:
        tmp = None

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if tmp is not None:
        subsequent_mask += tmp
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def resampling(source, tgt_tokens):
    pad_mask = tgt_tokens.eq(Constants.PAD)
    length = (1 - pad_mask).sum(-1)
    bsz, seq_len = tgt_tokens.shape

    all_idx = []
    scale = source.size(1) / length.float()
    for i in range(bsz):
        idx = (torch.arange(0, seq_len, device=tgt_tokens.device).float() * scale[i].repeat(seq_len)).long()
        max_idx = tgt_tokens.new(seq_len).fill_(source.size(1) - 1)
        idx = torch.where(idx < source.size(1), idx, max_idx)
        all_idx.append(idx)
    all_idx = torch.stack(all_idx, dim=0).unsqueeze(2).repeat(1, 1, source.size(2))
    return source.gather(1, all_idx)


class EmptyObject(object):
    def __init__(self):
        pass


def dict2obj(dict):
    obj = EmptyObject()
    obj.__dict__.update(dict)
    return obj


class BertDecoder(nn.Module):
    def __init__(self, config, embedding=None):
        super(BertDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)
        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False) if embedding is None else embedding

        self.layer = nn.ModuleList([BertLayer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])

        self.linear = nn.Linear(config.dim_hidden, config.vocab_size)

        self.layer_sim = nn.ModuleList([self_attention_layer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])
        self.pos_attention = config.pos_attention
        self.enhance_input = config.enhance_input
        self.watch = config.watch  # 0
        self.FCNet = config.is_FCNet  # 是否是FCNet
        self.BCNet = config.is_BCNet  # 是否是BCNet

        self.decoding_type = config.decoding_type

    def _init_embedding(self, weight, option={}, is_numpy=False):
        if is_numpy:
            self.embedding.word_embeddings.weight.data = 0
        else:
            self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', False):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def get_word_embeddings(self):
        return self.embedding.word_embeddings

    def set_word_embeddings(self, we):
        self.embedding.word_embeddings = we

    def forward(self, tgt_seq, enc_output=None, category=None, signals=None, tags=None, iteration=0, device=None, **kwargs):
        decoding_type = kwargs.get('decoding_type', self.decoding_type)
        output_attentions = kwargs.get('output_attentions', False)  # False

        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        all_attentions = ()

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        if decoding_type == 'NARFormer':
            slf_attn_mask = slf_attn_mask_keypad
        elif decoding_type == 'SelfMask':
            slf_attn_mask = slf_attn_mask_keypad
            seq_len = tgt_seq.size(1)

            diag =  torch.tril(torch.ones((seq_len, seq_len), device=slf_attn_mask.device, dtype=torch.uint8), diagonal=0) & \
                    torch.triu(torch.ones((seq_len, seq_len), device=slf_attn_mask.device, dtype=torch.uint8), diagonal=0)
            slf_attn_mask = (slf_attn_mask + diag).gt(0)

            # the i-th target can not see itself from the inputs
            '''
            tokens: <bos>   a       girl    is      singing <eos>
            target: a       girl    is      singing <eos>   ..
            '''
            #print(slf_attn_mask[0], slf_attn_mask.shape)
        else:
            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq, watch=self.watch)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            slf_attn_mask_sim = slf_attn_mask_keypad

        non_pad_mask = get_non_pad_mask(tgt_seq)
        src_seq = torch.ones(enc_output.size(0), enc_output.size(1)).to(enc_output.device)
        attend_to_enc_output_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        additional_feats = None
        if decoding_type == 'NARFormer':
            if self.enhance_input == 0:
                pass
            elif self.enhance_input == 1:
                additional_feats = resampling(enc_output, tgt_seq)
            elif self.enhance_input == 2:
                additional_feats = enc_output.mean(1).unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
            else:
                raise ValueError('enhance_input shoud be either 0, 1 or 2')

        if signals is not None:
            additional_feats = signals if additional_feats is None else (additional_feats + signals)

        if self.pos_attention:
            hidden_states, position_embeddings = self.embedding(tgt_seq, category=category)
        else:
            hidden_states = self.embedding(tgt_seq, additional_feats=additional_feats, category=category, tags=tags)
            position_embeddings = None

        # print("enc_output.shape：", enc_output.shape)
        # print('tgt_seq.shape:', tgt_seq.shape)
        # print('hidden_states.shape', hidden_states.shape)

        res = []
        res_FCNet = []
        for i, layer_module in enumerate(self.layer):
            if not i:
                input_ = hidden_states
            else:
                input_ = layer_outputs[0]# + hidden_states
                # print('here is second bertlayer :', input_.shape)

            layer_outputs = layer_module(
                input_,
                non_pad_mask=non_pad_mask,
                attention_mask=slf_attn_mask,  # 第一层mask self attetion 的mask
                enc_output=enc_output,
                attend_to_enc_output_mask=attend_to_enc_output_mask,  # 第二层mask inter attention 的mask
                position_embeddings=position_embeddings,
                word_embeddings=self.get_word_embeddings(),
                **kwargs
            )

            res.append(layer_outputs[0])
            # print('get decoder result:')
            # print(len(layer_outputs))
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
            embs = layer_outputs[1]

        res = [res[-1]]
        # tmp = torch.transpose(layer_outputs[0], 1, 2)
        # similarity_score = torch.bmm(layer_outputs[0], tmp)
        # sim_score = F.softmax(similarity_score, dim=2)

        # prediction_target = self.linear(layer_outputs[0]).max(2)[1]
        # print("prediction_target", prediction_target)
        # hidden_states_prediction = self.embedding(prediction_target, category=category)

        # non_pad_mask_sim = get_sim_non_pad_mask(tgt_seq)
        # key_pad_mask_sim = get_attn_sim_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        # slf_attn_mask_sim = key_pad_mask_sim

        if self.FCNet:
            for j, layer_module in enumerate(self.layer_sim):
                if not j:
                    input_ = layer_outputs[0]
                else:
                    input_ = layer_outputs_sim[0]  # + hidden_states

                layer_outputs_sim = layer_module(
                    input_,
                    non_pad_mask=non_pad_mask,
                    attention_mask=slf_attn_mask_sim,  # 第一层mask self attetion 的mask
                    enc_output=enc_output,
                    attend_to_enc_output_mask=attend_to_enc_output_mask,  # 第二层mask inter attention 的mask
                    position_embeddings=position_embeddings,
                    word_embeddings=self.get_word_embeddings(),
                    **kwargs
                )
                # layer_outputs_ = torch.bmm(sim_score, layer_outputs_sim[0])
                res_FCNet.append(layer_outputs_sim[0])
            res_FCNet = [res_FCNet[-1]]
            # print(layer_outputs_sim)

        if additional_feats == None and position_embeddings == None:
            if self.FCNet:
                outputs = (res, res_FCNet, embs,)
            else:
                outputs = (res, embs)
        else:
            if self.FCNet:
                outputs = (res, res_FCNet, embs,)
            else:
                outputs = (res, embs)

        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class Forward_Connected_BertDecoder(nn.Module):
    def __init__(self, config):
        super(Forward_Connected_BertDecoder, self).__init__()


    def forward_(self, hidden_states, enc_output,):
        pass

class BertDecoderDisentangled(nn.Module):
    def __init__(self, config):
        super(BertDecoderDisentangled, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)
        self.bert = BertDecoder(config)

    def get_word_embeddings(self):
        return self.bert.get_word_embeddings()

    def set_word_embeddings(self, we):
        self.bert.set_word_embeddings(we)

    def forward_(self, tgt_seq, enc_output, category, **kwargs):
        seq_probs, embs, *_ = self.bert(tgt_seq, enc_output, category, **kwargs)
        seq_probs = seq_probs[0]
        if len(_):
            return seq_probs, embs, _
        return seq_probs, embs

    def forward(self, tgt_seq, enc_output, category, **kwargs):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        if isinstance(tgt_seq, list):
            assert len(tgt_seq) == 2
            seq_probs1, _ = self.forward_(tgt_seq[0], enc_output, category, **kwargs)
            seq_probs2, embs = self.forward_(tgt_seq[1], enc_output, category, **kwargs)
            outputs = ([seq_probs1, seq_probs2],embs,)
        else:
            return self.forward_(tgt_seq, enc_output, category, **kwargs)
            # seq_probs, embs = self.forward_(tgt_seq, enc_output, category)
            # outputs = ([seq_probs],embs,)
        return outputs
