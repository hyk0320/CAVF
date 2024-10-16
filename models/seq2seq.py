import torch
import torch.nn as nn
from config import Constants
from models.bert import BertLayerNorm


class Seq2Seq(nn.Module):
    def __init__(self, 
                 opt,
                 preEncoder=None,
                 encoder=None,
                 joint_representation_learner=None,
                 auxiliary_task_predictor=None,
                 action_predictor=None,
                 decoder=None,
                 tgt_word_prj=None,
                 **kwargs
                 ):
        super(Seq2Seq, self).__init__()
        self.opt = opt
        self.preEncoder = preEncoder  # None
        self.encoder = encoder
        self.joint_representation_learner = joint_representation_learner
        self.auxiliary_task_predictor = auxiliary_task_predictor
        self.action_predictor = action_predictor
        self.decoder = decoder
        self.tgt_word_prj = tgt_word_prj

        if opt.get('tie_weights', False):
            self._tie_weights(opt['vocab_size'])

    def _tie_weights(self, vocab_size):
        word_embeddings = self.decoder.get_word_embeddings()
        self.tgt_word_prj.weight = word_embeddings.weight
        self.tgt_word_prj.bias = nn.Parameter(torch.zeros(vocab_size).float(), requires_grad=True)

    def encode(self, feats, gt_many_hot_verb=None, **kwargs):
        results = {}
        if self.opt.get('automatic_mask', False):  # False
            attention_mask = []
            for feat in feats:
                assert len(feat.shape) == 3
                attention_mask.append(feat.sum(-1).eq(0))
            results['attention_mask'] = attention_mask

        if self.preEncoder is not None:  # None
            feats = self.preEncoder(input_feats=feats)

        enc_output, enc_hidden, *attentions = self.encoder(feats)
        if len(attentions):
            results['encoder_attentions'] = attentions[0]

        if self.joint_representation_learner is not None:
            enc_output, enc_hidden = self.joint_representation_learner(enc_output, enc_hidden)

        if self.action_predictor is not None:
            # print("s2s的58行")
            results.update(self.action_predictor(enc_output=enc_output, gt_many_hot_verb=gt_many_hot_verb))

        if self.auxiliary_task_predictor is not None:
            auxiliary_results = self.auxiliary_task_predictor(
                enc_output=enc_output
                # action_prediction=results["verb"][1]
            )
            results.update(auxiliary_results)

        results['enc_output'] = enc_output
        results['enc_hidden'] = enc_hidden
            
        return results
    
    def prepare_inputs_for_decoder(self, encoder_outputs, category):
        input_keys_for_decoder = ['enc_output']
        if self.opt['decoding_type'] == 'LSTM':
            input_keys_for_decoder.append('enc_hidden')

        if self.opt.get('attribute', False) and self.opt.get('attribute_mode', 'none') != 'none':  # False
            input_keys_for_decoder += ['attr_probs', 'video2attr_raw_scores']
        
        inputs_for_decoder = {'category': category}
        for key in input_keys_for_decoder:
            inputs_for_decoder[key] = encoder_outputs[key]
        
        if isinstance(inputs_for_decoder['enc_output'], list):
            assert len(inputs_for_decoder['enc_output']) == 1
            inputs_for_decoder['enc_output'] = inputs_for_decoder['enc_output'][0]
        return inputs_for_decoder

    def forward(self, **kwargs):
        func_name = "forward_" + self.opt['decoding_type']
        return getattr(self, func_name, None)(kwargs)

    def forward_NARFormer(self, kwargs):
        if self.opt['is_NAR_decoder']:
            feats, tgt_tokens, mlm_tgt_tokens, full_sequence, category, device = map(
                lambda x: kwargs.get(x, None),
                ["feats", "tgt_tokens", "mlm_tgt_tokens", "full_sequence", "category", "device"]
            )
            print("hihi")
            # print('hi', mlm_tgt_tokens)
        elif self.opt['is_verb_decoder']:
            feats, verb, tgt_tokens, mlm_tgt_tokens, full_sequence, category, device, gt_many_hot_verb = map(
                lambda x: kwargs.get(x, None),
                ["feats", "verb", "tgt_tokens", "mlm_tgt_tokens", "full_sequence", "category", "device", "gt_many_hot_verb"]
            )
        elif self.opt['is_visual_decoder']:
            feats, tgt_tokens, category, device = map(
                lambda x: kwargs.get(x, None),
                ["feats", "tgt_tokens", "category", "device"]
            )

        results = self.encode(feats, gt_many_hot_verb=None)
        inputs_for_decoder = self.prepare_inputs_for_decoder(results, category)
        if self.opt['is_NAR_decoder']:
            # print("haha", tgt_tokens, mlm_tgt_tokens, full_sequence, self.tgt_word_prj != None, device != None)
            hidden_states, layer1, layer2, mlm_tgt_tokens_embeddings, embs, *_ = self.decoder(
                tgt_seq=tgt_tokens,
                mlm_tgt_tokens=mlm_tgt_tokens,
                full_sequence=full_sequence,
                tgt_word_prj=self.tgt_word_prj,
                **inputs_for_decoder,
                device=device
            )
        elif self.opt["is_verb_decoder"]:
            hidden_states, layer1, layer2, mlm_tgt_tokens_embeddings, embs, *_ = self.decoder(
                verb=verb,
                tgt_seq=tgt_tokens,
                mlm_tgt_tokens=mlm_tgt_tokens,
                full_sequence=full_sequence,
                tgt_word_prj=self.tgt_word_prj,
                **inputs_for_decoder,
                device=device,
                action_prediction=results["verb"][1]
            )
        elif self.opt['is_visual_decoder']:
            hidden_states, embs, *_ = self.decoder(
                tgt_seq=tgt_tokens,
                **inputs_for_decoder,
                device=device
            )

        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]

        tgt_word_logits = [self.tgt_word_prj(item) for item in hidden_states]
        tgt_word_logprobs = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits]

        results.update({
            Constants.mapping['lang'][0]: tgt_word_logprobs,
        })

        if self.opt['is_NAR_decoder']:
            results.update({
                # 'verb': results["verb"][0],
                'layer1': layer1,
                'layer2': layer2,
                'enc_output': inputs_for_decoder['enc_output'],
                'visual_words_embedding': mlm_tgt_tokens_embeddings,
            })
        elif self.opt['is_verb_decoder']:
            results.update({
                'verb': results["verb"][0],
                'layer1': layer1,
                'layer2': layer2,
                'enc_output': inputs_for_decoder['enc_output'],
                'visual_words_embedding': mlm_tgt_tokens_embeddings,
            })

        '''
            修改
        '''
        # a, b, c = LSTM_squence.chunk(3, dim=2)# batchsize * max_len * 512
        #
        # a_b_c_concat = torch.cat((a, b, c), dim=1)#bs * 90 * 512
        # LSTM_squence_logits = [self.tgt_word_prj(item) for item in a_b_c_concat]#bs * 90 * vocab_size
        # LSTM_squence_logprbs = [torch.log_softmax(item, dim=-1) for item in LSTM_squence_logits]#bs * 90 * vocab_size
        #
        # # new_LSTM_squence_logprbs = torch.tensor([torch.log_softmax(item, dim=-1).cpu().detach().numpy() for item in old_LSTM_squence_logprbs]).to(device)
        # # a_, b_, c_ = new_LSTM_squence_logprbs.chunk(3, dim=1)#bs * 30 * vocab_size
        # #
        # # LSTM_squence_logprbs = torch.cat((a_, b_, c_), dim=2)
        # # # .cpu().detach().numpy()
        # # LSTM_squence_logits1 = [self.tgt_word_prj(item) for item in a]
        # # LSTM_squence_logits2 = [self.tgt_word_prj(item) for item in b]
        # # LSTM_squence_logits3 = [self.tgt_word_prj(item) for item in c]
        # # LSTM_squence_logprbs1 = torch.tensor([torch.log_softmax(item, dim=-1).detach().cpu().numpy() for item in LSTM_squence_logits1]).to(device)
        # # LSTM_squence_logprbs2 = torch.tensor([torch.log_softmax(item, dim=-1).detach().cpu().numpy() for item in LSTM_squence_logits2]).to(device)
        # # LSTM_squence_logprbs3 = torch.tensor([torch.log_softmax(item, dim=-1).detach().cpu().numpy() for item in LSTM_squence_logits3]).to(device)
        # #
        # # LSTM_squence_logprbs = torch.cat((LSTM_squence_logprbs1, LSTM_squence_logprbs2, LSTM_squence_logprbs3), dim=2)#batch_size * max_len * 3*vocab_size
        #
        # results.update({
        #     'new_module_prediction': LSTM_squence_logprbs
        # })
        # ''''''

        return results

    def forward_ARFormer(self, kwargs):
        feats, tgt_tokens, category = map(
            lambda x: kwargs.get(x, None),
            ["feats", "tgt_tokens", "category"]
        )
        decoding_type = kwargs.get('decoding_type', self.opt['decoding_type'])
        pmlm_flag = (decoding_type == 'SelfMask')
        if pmlm_flag:
            tgt_tokens = [item[:, 1:] for item in tgt_tokens] if isinstance(tgt_tokens, list) else tgt_tokens[:, 1:]
        else:
            tgt_tokens = [item[:, :-1] for item in tgt_tokens] if isinstance(tgt_tokens, list) else tgt_tokens[:, :-1]

        results = self.encode(feats)
        inputs_for_decoder = self.prepare_inputs_for_decoder(results, category)
        if self.opt['is_FCNet']:
            hidden_states, hidden_states_sim, embs, *_ = self.decoder(
                tgt_seq=tgt_tokens,
                decoding_type=decoding_type,  # ARFormer
                output_attentions=kwargs.get('output_attentions', False),  # False
                **inputs_for_decoder
                )
        elif self.opt['is_BCNet']:
            # hidden_states, hidden_states_sim, hidden_states_visual, embs, *_ = self.decoder(
            #     tgt_seq=tgt_tokens,
            #     decoding_type=decoding_type,  # ARFormer
            #     output_attentions=kwargs.get('output_attentions', False),  # False
            #     **inputs_for_decoder
            # )
            hidden_states, hidden_states_visual, embs, *_ = self.decoder(
                tgt_seq=tgt_tokens,
                decoding_type=decoding_type,  # ARFormer
                output_attentions=kwargs.get('output_attentions', False),  # False
                **inputs_for_decoder
            )
        else:
            hidden_states, embs, *_ = self.decoder(
                tgt_seq=tgt_tokens,
                decoding_type=decoding_type,  # ARFormer
                output_attentions=kwargs.get('output_attentions', False),  # False
                **inputs_for_decoder
            )

        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]

        tgt_word_logits = [self.tgt_word_prj(item) for item in hidden_states]
        tgt_word_logprobs = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits]

        results.update({
            Constants.mapping['lang'][0]: tgt_word_logprobs,
        })

        if self.opt['is_FCNet']:
            if not isinstance(hidden_states_sim, list):
                hidden_states_sim = [hidden_states_sim]

            tgt_word_logits_sim = [self.tgt_word_prj(item) for item in hidden_states_sim]
            tgt_word_logprobs_sim = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits_sim]

            results.update({
                'FCNet_prediction': tgt_word_logprobs_sim
            })
        elif self.opt['is_BCNet']:
            # if not isinstance(hidden_states_sim, list):
            #     hidden_states_sim = [hidden_states_sim]

            if not isinstance(hidden_states_visual, list):
                hidden_states_visual = [hidden_states_visual]

            # tgt_word_logits_sim = [self.tgt_word_prj(item) for item in hidden_states_sim]
            # tgt_word_logprobs_sim = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits_sim]

            tgt_word_logits_visual = [self.tgt_word_prj(item) for item in hidden_states_visual]
            tgt_word_logprobs_visual = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits_visual]

            # results.update({
            #     'BCNet_prediction': tgt_word_logprobs_sim
            # })
            results.update({
                'BCNet_prediction_visual': tgt_word_logprobs_visual
            })
        return results
