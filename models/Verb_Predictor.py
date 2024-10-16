from config import Constants
import torch
import torch.nn as nn
from .bert import BertEmbeddings, BertLayer
from .self_attention import self_attention_layer
from .NAR_self_attention import cross_attention_layer_1
from torch.nn import Parameter
import torch.nn.functional as F

# from models import LSTM_squence, LSTM_squence_v2

__all__ = ('BertDecoder', 'BertDecoderDisentangled')

def change_to_many_hot_representation(out):
    assert out.dim() == 2
    max_probs, idx = out.max(dim=-1)
    print("这里是VP的第17行：", max_probs.size(), max_probs, idx.size(), idx)
    # new_out = torch.FloatTensor([[0]*out.size(1)]*out.size(0))
    # print("这里是Verb_Predictor的17行：", idx.size())
    # for _ in range(out.size(0)):
    #     new_out[_][idx[_]] = 1
    # return out

def get_non_pad_mask(seq):
    # print(seq.shape)
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_non_mask_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.MASK).type(torch.float).unsqueeze(-1)

def get_non_mask_and_pad_mask(seq):
    assert seq.dim() == 2
    not_equal_pad = seq.ne(Constants.PAD)
    not_equal_mask = seq.ne(Constants.MASK)
    not_equal_mask_and_pad = not_equal_pad * not_equal_mask
    return not_equal_mask_and_pad.type(torch.float).unsqueeze(-1)

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

def get_attn_key_pad_and_mask_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask_1 = seq_k.eq(Constants.PAD)
    padding_mask_2 = seq_k.eq(Constants.MASK)

    zero = torch.zeros_like(padding_mask_1)
    one = torch.ones_like(padding_mask_1)

    padding_mask = torch.where((padding_mask_1 + padding_mask_2) > 0, one, zero)

    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

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
        # print(config.pos_attention, config.num_hidden_layers_decoder)
        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False) if embedding is None else embedding

        # if getattr(config, "is_verb_decoder", None):
        #     #   verb predictor
        #     self.verb_pre = nn.Sequential(
        #         nn.Linear(config.dim_hidden, config.dim_hidden),
        #         nn.ReLU(),
        #         nn.Dropout(config.hidden_dropout_prob),
        #         nn.Linear(config.dim_hidden, config.verb_num),
        #         nn.Sigmoid()
        #     )
        #     self.proj = nn.Linear(config.verb_num, config.dim_hidden)

        self.layer = nn.ModuleList([BertLayer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])

        self.linear = nn.Linear(config.dim_hidden, config.vocab_size)

        self.layer_visual_text = nn.ModuleList([cross_attention_layer_1(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])

        # self.layer_sim = nn.ModuleList([self_attention_layer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])

        # self.layer_visual = nn.ModuleList([self_attention_layer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])

        self.pos_attention = config.pos_attention  # False
        self.enhance_input = config.enhance_input
        self.watch = config.watch  # 0
        # self.FCNet = config.is_FCNet  # 是否是FCNet
        # self.BCNet = config.is_BCNet  # 是否是BCNet

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

    def forward(self, verb, tgt_seq, enc_output=None, mlm_tgt_tokens=None, full_sequence=None, tgt_word_prj=None, category=None, action_prediction=None, signals=None, tags=None, iteration=0, device=None, **kwargs):
        """
            args:
                verb: True or False for need verb input?
                verb_gt_many_hot: verb labels like [0,0,0,1,1,0,0,0,...]
                tgt_seq: inputs of decoder [6, 12, [mask], 67, ..., [pad]] or [[vis], [vis], [vis], [vis], ..., [vis]]
                mlm_tgt_tokens: labels of decoder [[pad], [pad], 3, [pad], ..., [pad]] or [[pad], [pad], ..., 3, [pad], [pad]]
                full_labels: full sequence of input text [6, 12, 3, 67, ..., [pad]]
                [mask]: 4
                [pad]: 0
        """

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
            if mlm_tgt_tokens is not None:
                # print(full_sequence)
                mlm_hidden_states, position_embeddings = self.embedding(mlm_tgt_tokens, category=category)
        else:
            hidden_states = self.embedding(tgt_seq, additional_feats=additional_feats, category=category, tags=tags)
            position_embeddings = None

        # print("enc_output.shape：", enc_output.shape)
        # print('tgt_seq.shape:', tgt_seq.shape)
        # print('hidden_states.shape', hidden_states.shape)

        # print(verb, gt_many_hot_verb.size())
        if verb is True:
            # if isinstance(enc_output, list):
            #     assert len(enc_output) == 1
            # enc_output = enc_output[0]
            # assert len(enc_output.shape) == 3
            # if gt_many_hot_verb is not None:
            #     print("gt_many_hot_verb:", gt_many_hot_verb)
            #     out = self.verb_pre(enc_output.mean(1))  # bs * verb_num
            #     print("看看out的数据是怎么样的：", out.size(), out)
            #     out1 = self.proj(gt_many_hot_verb)   # bs * dim_size
            #     print("out1的数据是怎么样的：", out1.size(), out1)
            #     out2 = out1.unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
            #     print("out2的数据：", out2.size())
            #     hidden_states = hidden_states + out2  # bs * 30 * dim_size
            # else:
            #     print("这里是Verb_Predictor的264行：")
            #     out = self.verb_pre(enc_output.mean(1))  # bs * verb_num
            #     print("out这里是去除TF训练方式的做法：", out)
            #     change_to_many_hot_representation(out)
            #     # out = change_to_many_hot_representation(out)
            #     out1 = self.proj(out)   # bs * dim_size
            #     out2 = out1.unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
            #     hidden_states = hidden_states + out2  # bs * 30 * dim_size
            # print(action_prediction.size(), hidden_states.size(), tgt_seq.size())
            if action_prediction.size(0) * 6 == hidden_states.size(0):
                action_prediction1 = action_prediction.unsqueeze(1).repeat(6, tgt_seq.size(1), 1)
            else:
                action_prediction1 = action_prediction.unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
            hidden_states = hidden_states + action_prediction1

        if mlm_tgt_tokens is not None:
            # print('mlm:', mlm_tgt_tokens, mlm_tgt_tokens.shape)
            non_pad_mask_ = get_non_pad_mask(src_seq)
            attention_mask = get_attn_key_pad_and_mask_mask(seq_k=mlm_tgt_tokens, seq_q=src_seq)
            # attention_mask = get_attn_key_pad_mask(seq_k=mlm_tgt_tokens, seq_q=src_seq)
            # print(tgt_seq, tgt_seq.shape)
            # print(tgt_seq+mlm_tgt_tokens, (tgt_seq+mlm_tgt_tokens).shape)
        if mlm_tgt_tokens is not None:
            mask_ = get_non_pad_mask(mlm_tgt_tokens)
            # mask_ = get_non_pad_mask(mlm_tgt_tokens)
            new_mlm_hidden_states = mlm_hidden_states * mask_
            for j, layer_module in enumerate(self.layer_visual_text):
                if not j:
                    input_ = enc_output
                    # input_ = enc_output
                layer_outputs_cross_attention1 = layer_module(
                    input_,
                    non_pad_mask=non_pad_mask_,
                    enc_output=new_mlm_hidden_states,
                    attend_to_enc_output_mask=attention_mask,
                    sim_or_visual=0,
                    **kwargs
                )

        res = []
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

        if mlm_tgt_tokens is not None:
            # voca_tgt = tgt_word_prj(layer_outputs[0]).max(2)[1]
            # mask = get_non_pad_mask(mlm_tgt_tokens).squeeze(-1)
            mask = get_non_pad_mask(mlm_tgt_tokens)
            new_voca_tgt = (layer_outputs[0] + position_embeddings) * mask
            # print(new_voca_tgt)

            # mask_ = get_non_mask_mask(tgt_seq).squeeze(-1)
            # new_tgt_seq = tgt_seq * mask_
            # print(new_tgt_seq)
            # print('prediction:', voca_tgt, voca_tgt.shape)
            # voca_tgt_hidden_states, _ = self.embedding((new_voca_tgt+new_tgt_seq).type(torch.long), category=category)
            # voca_tgt_hidden_states, _ = self.embedding(new_voca_tgt.type(torch.long), category=category)
            for k, layer_module in enumerate(self.layer_visual_text):
                if not k:
                    input_ = enc_output
                    # input_ = enc_output
                layer_outputs_cross_attention2 = layer_module(
                    input_,
                    non_pad_mask=non_pad_mask_,
                    enc_output=new_voca_tgt,
                    attend_to_enc_output_mask=attention_mask,
                    sim_or_visual=0,
                    **kwargs
                )

        if additional_feats == None and position_embeddings == None:
            if mlm_tgt_tokens is not None:
                outputs = (res, layer_outputs_cross_attention1[0], layer_outputs_cross_attention2[0], new_mlm_hidden_states, embs)
            else:
                outputs = (res, embs)
        else:
            if mlm_tgt_tokens is not None:
                outputs = (res, layer_outputs_cross_attention1[0], layer_outputs_cross_attention2[0], new_mlm_hidden_states, embs)
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
        print(config)
        self.bert = BertDecoder(config)

    def get_word_embeddings(self):
        return self.bert.get_word_embeddings()

    def set_word_embeddings(self, we):
        self.bert.set_word_embeddings(we)

    def forward_(self, verb, tgt_seq, enc_output, mlm_tgt_tokens=None, full_sequence=None, tgt_word_prj=None, category=None, action_prediction=None, **kwargs):
        if mlm_tgt_tokens is not None:  # train
            # print("VP403:", action_prediction)
            seq_probs, layer1, layer2, mlm_tgt_tokens_embeddings, embs, *_ = self.bert(verb, tgt_seq, enc_output, mlm_tgt_tokens, full_sequence, tgt_word_prj, category, action_prediction, **kwargs)
            seq_probs = seq_probs[0]
            if len(_):
                return seq_probs, layer1, layer2, mlm_tgt_tokens_embeddings, embs, _
            return seq_probs, layer1, layer2, mlm_tgt_tokens_embeddings, embs
        else:  # test
            seq_probs, embs, *_ = self.bert(verb, tgt_seq, enc_output, mlm_tgt_tokens, full_sequence, tgt_word_prj, category, action_prediction, **kwargs)
            seq_probs = seq_probs[0]
            if len(_):
                return seq_probs, embs, _
            return seq_probs, embs

    def forward(self, verb, tgt_seq, enc_output, mlm_tgt_tokens=None, full_sequence=None, tgt_word_prj=None, category=None, action_prediction=None, **kwargs):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        if isinstance(tgt_seq, list):   # visual words?   train
            assert len(tgt_seq) == 2
            if mlm_tgt_tokens is not None:
                # print("visual words labels:", mlm_tgt_tokens[0])
                # print("mlm words labels:", mlm_tgt_tokens[1])
                # print("hhhhhh:", verb)
                seq_probs1, layer3, layer4, mlm_tgt_tokens_embeddings,  *_ = self.forward_(verb, tgt_seq[0], enc_output, mlm_tgt_tokens[0], full_sequence, tgt_word_prj, category, action_prediction, **kwargs)
                # print("noun , verb:", tgt_seq[0][0], tgt_seq[0].shape, mlm_tgt_tokens[0], mlm_tgt_tokens[0].shape)
                seq_probs2, embs = self.forward_(True, tgt_seq[1], enc_output, None, full_sequence, tgt_word_prj, category, action_prediction, **kwargs)
                # print("bert data:", tgt_seq[1], tgt_seq[1].shape, mlm_tgt_tokens[1], mlm_tgt_tokens[1].shape)
                # outputs = ([seq_probs1, seq_probs2], [layer1, layer3], [layer2, layer4], embs,)
                outputs = ([seq_probs1, seq_probs2], [layer3], [layer4], mlm_tgt_tokens_embeddings, embs,)  # 对视觉词进行对齐
            else:
                print("here is one")
                seq_probs1, _ = self.forward_(tgt_seq[0], enc_output, category, **kwargs)
                seq_probs2, embs = self.forward_(tgt_seq[1], enc_output, category, **kwargs)
                outputs = ([seq_probs1, seq_probs2], embs,)
        else:  # test
            # print("here is two")
            if mlm_tgt_tokens is None:
                # print("VP440:", action_prediction)
                return self.forward_(verb, tgt_seq, enc_output, mlm_tgt_tokens, full_sequence, tgt_word_prj, category, action_prediction, **kwargs)
            else:
                print("mlm_tgt_tokens is not None!")
            # seq_probs, embs = self.forward_(tgt_seq, enc_output, category)
            # outputs = ([seq_probs],embs,)
        return outputs
