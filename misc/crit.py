from .logger import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Constants
from torch.autograd import Variable
import math
from collections import defaultdict


class CritBase(nn.Module):
    def __init__(self, opt, crit_name, weights=1.0, batch_mean=True):
        super(CritBase, self).__init__()
        assert crit_name in Constants.mapping.keys()
        self.keys = Constants.mapping[crit_name]
        self.weights = weights
        self.batch_mean = batch_mean
        self.opt = opt

    def _step(self, *inputs):
        raise NotImplementedError()

    def caculate_new_module_loss(self, *input):
        raise NotImplementedError()

    def forward(self, kwargs):
        sources1, sources2, *others = [kwargs[key] for key in self.keys]

        if 'new_module_prediction' in kwargs.keys():
            source3 = kwargs['new_module_prediction']#必须要是tensor，shape为bs * max_len * 3*vocab_size
            # print(kwargs.keys())
            source4 = kwargs['new_module_labels']
        elif 'FCNet_prediction' in kwargs.keys():
            source5 = kwargs['FCNet_prediction']
        elif 'BCNet_prediction_visual' in kwargs.keys():
            # sources6 = kwargs['BCNet_prediction']
            # sources7 = kwargs['BCNet_prediction_visual']
            sources6 = kwargs['BCNet_prediction_visual']
        elif 'layer1' in kwargs.keys():
            sources8 = kwargs['layer1']
            sources9 = kwargs['layer2']
            sources10 = kwargs['enc_output']
            sources11 = kwargs['visual_words_embedding']
            # sources10 = torch.softmax(sources10, dim=-1)

            # sources12 = kwargs['verb']
            # sources13 = kwargs['verb_label']
            # print('crit45', sources12, sources13)
        else:
            pass


        if not isinstance(sources1, list):
            assert type(sources1) == torch.Tensor
            sources1 = [sources1]
        
        if not isinstance(sources2, list):
            assert type(sources2) == torch.Tensor
            sources2 = [sources2] * len(sources1)
        else:
            assert len(sources1) == len(sources2)

        if not isinstance(self.weights, list):
            self.weights = [self.weights] * len(sources1)

        assert len(sources1) == len(self.weights)

        loss = None
        FCNet_loss = None
        BCNet_loss1 = None
        BCNet_loss2 = None
        NAR_loss = None
        dinominator = sources1[0].size(0) if self.batch_mean else 1.0  # bs
        # print('dinominator:', dinominator)
        if 'new_module_prediction' in kwargs.keys():
            dinominator1 = len(source3) if self.batch_mean else 1.0
            # print('dinominator1:', dinominator1)

        if 'FCNet_prediction' in kwargs.keys():
            if not isinstance(source5, list):
                assert type(source5) == torch.Tensor
                source5 = [source5]

        if 'BCNet_prediction_visual' in kwargs.keys():
            if not isinstance(sources6, list):
                # assert (type(sources6) == torch.Tensor) and (type(sources7) == torch.Tensor)
                assert (type(sources6) == torch.Tensor)
                sources6 = [sources6]
                # sources7 = [sources7]

        if 'layer1' in kwargs.keys():
            if not isinstance(sources8, list):
                # print(type(sources8))
                assert (type(sources8) == torch.Tensor) and (type(sources9) == torch.Tensor) and (type(sources10) == torch.Tensor)
                sources8 = [sources8]
                sources9 = [sources9]
                sources10 = [sources10]
                # print(sources10[0].shape)

        if 'verb' in kwargs.keys():
            if not isinstance(sources8, list):
                assert (type(sources8) == torch.Tensor) and (type(sources9) == torch.Tensor) and (type(sources10) == torch.Tensor)
                sources8 = [sources8]
                sources9 = [sources9]
                sources10 = [sources10]
                sources12 = [sources12]
                sources13 = [sources13]

        for i, (weight, src1, src2) in enumerate(zip(self.weights, sources1, sources2)):
            if loss is None:
                loss = weight * self._step(i, src1, src2, *others) / dinominator
            else:
                loss = loss + weight * self._step(i, src1, src2, *others) / dinominator

        if 'new_module_prediction' in kwargs.keys():
            new_module_loss = self.caculate_new_module_loss(source3, source4)/dinominator1
            loss = loss + new_module_loss * 0.1
            return new_module_loss, loss, dinominator

        elif "FCNet_prediction" in kwargs.keys():
            # print(sources2[0].size())
            tmp = sources1[0].max(2)[1]
            # print(tmp.size())
            # tmp = source5[0].max(2)[1]
            # for j, (weight, src5, src2) in enumerate(zip(self.weights, [tmp], sources2)):
            for j, (weight, src5, src2) in enumerate(zip(self.weights, source5, [tmp])):
                if FCNet_loss is None:
                    FCNet_loss = weight * self._step(j, src5, src2, *others) / dinominator
                else:
                    FCNet_loss = FCNet_loss + weight * self._step(j, src5, src2, *others) / dinominator
            return FCNet_loss, (loss+FCNet_loss*self.opt.get('FCNet_loss_weight', 1)), dinominator
            pass  # caculate last trasnformer layers' loss

        elif 'BCNet_prediction_visual' in kwargs.keys():
            for k1, (weight, src6, src2) in enumerate(zip(self.weights, sources6, sources2)):
                if BCNet_loss1 is None:
                    BCNet_loss1 = weight * self._step(k1, src6, src2, *others) / dinominator
                else:
                    BCNet_loss1 = BCNet_loss1 + weight * self._step(k1, src6, src2, *others) / dinominator
            # for k2, (weight, src7, src2) in enumerate(zip(self.weights, sources7, sources2)):
            #     if BCNet_loss2 is None:
            #         BCNet_loss2 = weight * self._step(k2, src7, src2, *others) / dinominator
            #     else:
            #         BCNet_loss2 = BCNet_loss2 + weight * self._step(k1, src7, src2, *others) / dinominator
            # return BCNet_loss1, BCNet_loss2, (loss + BCNet_loss1 * self.opt.get('BCNet_loss1_weight', 1) + BCNet_loss2 * self.opt.get('BCNet_loss2_weight', 1)), dinominator
            return BCNet_loss1, (loss + BCNet_loss1 * self.opt.get('BCNet_loss1_weight', 1)), dinominator

        elif 'layer1' in kwargs.keys():
            for m, (src8, src9) in enumerate(zip(sources8, sources9)):
                if NAR_loss is None:
                    NAR_loss = self.caculate_KL_loss(src8, src9)/dinominator
                else:
                    NAR_loss = NAR_loss + self.caculate_KL_loss(src8, src9)/dinominator

            for n, (src8, src10) in enumerate(zip(sources8, [sources10])):    # (DSA loss) visual alignment : output of gt
                NAR_loss = NAR_loss + self.caculate_cos_loss(src8, src10)/dinominator

            for k, (src8, src10) in enumerate(zip(sources8, [sources10])):
                NAR_loss = NAR_loss + self.caculate_cos_1_loss(src8, src10)/dinominator

            # for index1, (src9, src10) in enumerate(zip(sources9, [sources10])):    # (DSA loss) visual alignment : output of prediction
            #     NAR_loss = NAR_loss + self.caculate_cos_loss(src9, src10)/dinominator + self.caculate_cos_1_loss(src9, src10)/dinominator


            NAR_loss = NAR_loss * self.opt.get('NAR_loss_weight', 1)

            if self.opt.get('is_verb_decoder', False):
                print('crit168',self.opt.get('is_verb_decoder', False))
                for j, (src12, src13) in enumerate(zip([sources12], [sources13])):  # verb constraints bs * verb_num
                    NAR_loss = NAR_loss + (self.caculate_exp_loss(src12, src13)/dinominator) * self.opt.get('verb_loss_weight', 1)

                # for n, (src10, src11) in enumerate(zip([sources11], [sources10])):
            #     NAR_loss = NAR_loss + self.caculate_cosine_loss(src10, src11)/dinominator

            return NAR_loss, (loss + NAR_loss), dinominator
            # return NAR_loss, (loss + NAR_loss * self.opt.get('NAR_loss_weight', 1)), dinominator
        else:
            return loss, dinominator


class LanguageGeneration(CritBase):
    def __init__(self, opt, crit_name, weights=1.0, batch_mean=True):
        visual_word_generation = opt.get('visual_word_generation', False)
        if visual_word_generation:
            weights = opt.get('nv_weights', [0.8, 1.0])
        super().__init__(opt, crit_name, weights, batch_mean)
        self.loss_fn = nn.NLLLoss(reduce=False)
        self.ignore_index = Constants.PAD
        self.num_word_acc = 2 if visual_word_generation else 1
        self.visual_word_generation = visual_word_generation
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.kl_loss = nn.KLDivLoss(reduction='sum')
        self.XE_loss = nn.CrossEntropyLoss(reduction="sum")
        self.softmax_1 = nn.Softmax(dim=-1)
        self.softmax_2 = nn.Softmax(dim=1)
        self.softmax_3 = nn.Softmax(dim=0)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.BCELoss = nn.BCELoss(reduction='none')

    def caculate_verb_loss(self, pre, label):
        """
            arg: pre bs * verb_num
                 label bs * verb_num
        """
        # print(pre.size(), label.size())
        assert pre.size() == label.size()

        loss = self.l2_loss(pre, label)
        return loss

    def caculate_cosine_loss(self, mlm_tgt_seq_embeddings, enc_output):
        """
            mean word- mean visual frame alignment
            args:
                mlm_tgt_seq_embeddings:    [bs, ?, 512]
                enc_output:     [bs, 2 * frame_len, 512]
        """
        bs, frame_len_2, d_m = enc_output.size()
        _, len_seq, __ = mlm_tgt_seq_embeddings.size()

        align_loss = 0

        new_mlm_tgt_seq_embeddings = torch.sum(mlm_tgt_seq_embeddings, dim=1)/len_seq   # bs * 512
        new_enc_output = torch.sum(enc_output, dim=1)/frame_len_2   # bs * 512

        # print(new_mlm_tgt_seq_embeddings.size())
        # print(new_enc_output.permute(1, 0).size())
        cosine_score_1 = torch.mm(new_mlm_tgt_seq_embeddings, new_enc_output.permute(1, 0))  # bs * bs
        tmp_1 = torch.norm(new_mlm_tgt_seq_embeddings, dim=-1).unsqueeze(-1)  # bs * 1
        tmp_2 = torch.norm(new_enc_output, dim=-1).unsqueeze(-1)    # bs * 1

        cosine_score_2 = torch.mm(tmp_1, tmp_2.permute(1, 0))   # bs * bs

        cosine_score = cosine_score_1/cosine_score_2    # bs * bs

        for i in range(bs):
            # clone_cosine_score = cosine_score.clone()
            clone_cosine_score = cosine_score
            align_loss = align_loss - torch.sum(torch.log(self.softmax_1(clone_cosine_score))[i, i]) - torch.sum(torch.log(torch.log(self.softmax_2(clone_cosine_score))[i, i]))

        return align_loss

    def caculate_cos_loss(self, result_of_cross, enc_output):
        """
            alignment of reconstruction visual features by words
            args：
                result_of_cross:  [bs, 2 * frame_len, 512]
                enc_output:  [bs, 2 * frame_len, 512]
        """
        assert result_of_cross.size() == enc_output.size()
        # print(result_of_cross.size(), enc_output.size())
        bs, frame_len_2, _ = result_of_cross.shape
        # non_mask = t.ne()
        device = torch.device('cuda')
        # cosine_score = torch.Tensor([[0 for _ in range(bs)] for __ in range(bs)]).to(device)
        cosine_score = torch.zeros([bs, bs])
        # softmax = torch.zeros([bs])
        align_loss = 0

        # for i in range(bs):
        #     cosine_score[i, i] = torch.cosine_similarity(result_of_cross[i, :].view(-1), enc_output[i, :].view(-1), dim=-1)
        #     for j in range(bs):
        #         if i != j:
        #             cosine_score[i, j] = torch.cosine_similarity(result_of_cross[i, :].view(-1), enc_output[j, :].view(-1), dim=-1)
        #
        # new_cosine_score = cosine_score.clone()
        # for i in range(bs):
        #     softmax = F.log_softmax(new_cosine_score[i, i], dim=-1)
        #     # print(softmax.shape)
        #     align_loss = align_loss - softmax[i]
        #
        # return align_loss
        # print(result_of_cross.contiguous().view(bs, -1).size(), enc_output.contiguous().view(bs, -1).permute(1, 0).size())
        cosine_score_1 = torch.bmm(result_of_cross, enc_output.permute(0, 2, 1))
        tmp_1 = torch.norm(result_of_cross, dim=-1).unsqueeze(-1)  # bs * 16 * 1
        tmp_2 = torch.norm(enc_output, dim=-1).unsqueeze(-1)
        # print(tmp_2.size())
        cosine_score_2 = torch.bmm(tmp_1, tmp_2.permute(0, 2, 1))   # bs * 16 * 16

        cosine_score = cosine_score_1 / cosine_score_2  # bs * 16 * 16

        for i in range(frame_len_2):
            new_cosine_score = cosine_score.clone()
            align_loss = align_loss - torch.sum(torch.log(self.softmax_1(new_cosine_score))[:, i, i]) - torch.sum(torch.log(self.softmax_2(new_cosine_score))[:, i, i])

        return align_loss

        # cos_similarity = torch.sum(torch.cosine_similarity(v, t, dim=-1))

    def caculate_cos_1_loss(self, result_of_cross, enc_output):
        """
            alignment of reconstruction visual features by words
            args：
                result_of_cross:  [bs, 2 * frame_len, 512]
                enc_output:  [bs, 2 * frame_len, 512]
        """
        assert result_of_cross.size() == enc_output.size()
        bs, frame_len_2, _ = result_of_cross.shape

        align_loss = 0

        new_result_of_cross = result_of_cross.view(bs, -1)  # bs * (16*512)
        new_enc_output = enc_output.view(bs, -1)    # bs * (16*512)

        cosine_score_1 = torch.mm(new_result_of_cross, new_enc_output.permute(1, 0))  # bs * bs
        tmp_1 = torch.norm(new_result_of_cross, dim=-1).unsqueeze(-1)   # bs * 1
        tmp_2 = torch.norm(new_enc_output, dim=-1).unsqueeze(-1)    # bs * 1

        # print(tmp_2.shape)
        # raise EOFError

        cosine_score_2 = torch.mm(tmp_1, tmp_2.permute(1, 0))

        cosine_score = cosine_score_1 / cosine_score_2

        for i in range(bs):
            new_cosine_score = cosine_score.clone()
            align_loss = align_loss - torch.log(self.softmax_3(new_cosine_score))[i, i] - torch.log(self.softmax_1(new_cosine_score))[i, i]

        return align_loss



    def caculate_KL_loss(self, layer1, layer2):
        """
            args:
                layer1: [bs, 2 * frame_len, 512]
                layer2: [bs, 2 * frame_len, 512]
        """
        # print(layer1.size(), layer2.size())
        assert layer1.size() == layer2.size()

        loss = self.kl_loss(torch.log(self.softmax_1(layer1)), self.softmax_1(layer2))
        return loss

    def caculate_KL_loss2(self, prediction, label):
        """
            args:
                prediction: [bs, voc_len]
                label: [bs, voc_len]
        """
        assert prediction.size() == label.size()
        # 下面这种写法是去保证学习稀疏化向量的有效途径，具体参考length_predictor
        label = label / torch.sum(label, dim=-1).unsqueeze(1)
        # assert label.size() == prediction.size()
        print("crit的336行：", prediction.size(), prediction, label.size(), label)
        max_probs, idx = prediction.max(dim=-1)
        max_probs1, idx1 = label.max(dim=-1)
        print("crit这里是339行：", max_probs.size(), max_probs, idx.size(), idx, max_probs1.size(), max_probs1, idx1.size(), idx1)

        # loss = sum(torch.log(torch.exp(prediction*label) + 1).squeeze())
        loss = self.kl_loss(self.softmax_1(prediction).log(), label)
        # loss = self.BCELoss(prediction, label)
        print("crit的342行：", loss)
        return loss
        # print(prediction.size(), label.size())
        # print(label.size())
        # print(prediction.size())
        # new_label = label.nonzero()  # bs * k * 1
        # print("new_label1:", new_label.size())
        # new_label = new_label.squeeze(-1)
        # print("new_label2:", new_label.size())
        # loss = self.XE_loss(prediction, new_label.type(torch.long))
        # return loss

    # o2na里面采用的分类预测损失函数
    def caculate_exp_loss(self, prediction, label):
        """
            args:
                prediction: [bs, voc_len]
                label: [bs, voc_len]
        """
        assert prediction.size() == label.size()
        # print("crit的336行：", prediction.size(), prediction, label.size(), label)
        max_probs, idx = prediction.max(dim=-1)
        max_probs1, idx1 = label.max(dim=-1)
        # print("crit这里是339行：", max_probs.size(), max_probs, idx.size(), idx, max_probs1.size(), max_probs1, idx1.size(), idx1)

        loss = torch.sum(torch.log(torch.exp(-prediction*label) + 1).squeeze())
        # print("crit的342行：", loss)
        return loss

    # 多标签分类抑制长尾效应
    def caculate_multiLabelMargin_loss(self, prediction, label):
        """
            args:
                prediction: [bs, voc_len]
                label: [bs, voc_len]
        """
        assert prediction.size() == label.size()

    # 多标签分类抑制长尾效应
    def caculate_focal_loss(self, prediction, label):
        """
            loss = -alfa_t * (1-p_t)**gama * log(p_t)
            args:
                prediction: [bs, voc_len]
                label: [bs, voc_len]
        """
        assert prediction.size() == label.size()

        print("crit的336行：", prediction.size(), prediction, label.size(), label)
        max_probs, idx = prediction.max(dim=-1)
        max_probs1, idx1 = label.max(dim=-1)
        print("crit这里是339行：", max_probs.size(), max_probs, idx.size(), idx, max_probs1.size(), max_probs1, idx1.size(), idx1)

        BCELoss = self.BCELoss(prediction, label)

        pt = torch.exp(-BCELoss)

        loss_tensor = (1-pt)**2 * BCELoss

        p_loss = torch.sum(loss_tensor)

        print("crit的407行：", p_loss)

        return p_loss

    def caculate_l2_loss(self, layer1, layer2):
        """
            args:
                layer1: [bs, 2 * frame_len, 512]
                layer2: [bs, 2 * frame_len, 512]
        """
        assert layer1.size() == layer2.size()

        loss = self.l2_loss(layer1, layer2)
        return loss

    def caculate_new_module_loss(self, new_module_prediction, new_module_labels):
        '''
            args:
                new_module_prediction:[batch_size, 3*seq_len, vocab_size]
                new_module_labels:[batch_size, 3*seq_len]
        '''
        # print(new_module_prediction, len(new_module_prediction))
        # print('#####')
        # print(type(new_module_labels))
        # device = torch.device('cuda')
        new_module_prediction = Variable(torch.stack(tuple(new_module_prediction), 0), requires_grad=True)
        # print("123456789")
        # print(type(new_module_prediction))
        # print(type(new_module_labels))
        # print(new_module_prediction.device)
        # print(new_module_labels.device)
        # print('new_module_prediction.shape:', new_module_prediction.size())
        # print('new_module_labels.shape:', new_module_labels.shape)

        pre = new_module_prediction.contiguous().view(-1, new_module_prediction.size(2))
        lab = new_module_labels.contiguous().view(-1)

        print('pre:', pre.size())
        print('lab:', lab.size())

        new_module_loss = self.loss_fn(pre, lab)

        print('new_module_loss:', new_module_loss.size(), new_module_loss)

        if self.ignore_index is not None:
            mask = lab.ne(self.ignore_index).float()
            print('mask:', mask.size(), mask)
            sum = torch.sum(new_module_loss * mask)
            return sum
        else:
            print('no pad in crit.py')
            raise NotImplementedError

        # a_p, b_p, c_p = new_module_prediction1.chunk(3, dim=1)
        # # print(new_module_prediction.shape)
        # # print(new_module_labels)
        # # print(new_module_labels.shape)
        # a_g, b_g, c_g = new_module_labels.chunk(3, dim=1)
        #
        # a_p = a_p.contiguous().view(-1, a_p.size(2))
        # b_p = b_p.contiguous().view(-1, b_p.size(2))
        # c_p = c_p.contiguous().view(-1, c_p.size(2))
        #
        # a_g = a_g.contiguous().view(-1)
        # b_g = b_g.contiguous().view(-1)
        # c_g = c_g.contiguous().view(-1)
        #
        # new_module_loss1 = self.loss_fn(a_p, a_g)
        # new_module_loss2 = self.loss_fn(b_p, b_g)
        # new_module_loss3 = self.loss_fn(c_p, c_g)
        #
        # if self.ignore_index is not None:
        #     mask1 = a_g.ne(self.ignore_index).float()
        #     mask2 = b_g.ne(self.ignore_index).float()
        #     mask3 = c_g.ne(self.ignore_index).float()
        #
        #     sum1 = torch.sum(new_module_loss1 * mask1)
        #     sum2 = torch.sum(new_module_loss2 * mask2)
        #     sum3 = torch.sum(new_module_loss3 * mask3)
        #     return (sum1 + sum2 + sum3)
        # else:
        #     print('no pad in crit.py')
        #     raise NotImplementedError

    def _step(self, index_indicator, tgt_word_logprobs, tgt_word_labels, *others):
        """
            args:
                tgt_word_logprobs: [batch_size, seq_len, vocab_size]
                tgt_word_labels: [batch_size, seq_len]
        """
        assert not len(others)
        assert tgt_word_logprobs.size(1) == tgt_word_labels.size(1)

        # calculate the top-1 accuracy of the generated words
        self.calculate_word_acc(index_indicator, tgt_word_logprobs, tgt_word_labels)
        # calculate the perplexity of the generated words
        self.calculate_perplexity(index_indicator, tgt_word_logprobs, tgt_word_labels)

        tgt_word_logprobs = tgt_word_logprobs.contiguous().view(-1, tgt_word_logprobs.size(2))
        # print("tgt_word_logprobs.size():", tgt_word_logprobs.size())
        tgt_word_labels = tgt_word_labels.contiguous().view(-1)
        # print("tgt_word_labels.size():", tgt_word_labels.size())
        loss = self.loss_fn(tgt_word_logprobs, tgt_word_labels)

        if self.ignore_index is not None:
            mask = tgt_word_labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask)
        else:
            return torch.sum(loss)
    
    def calculate_word_acc(self, index_indicator, preds, gts):
        ind = gts.ne(Constants.PAD)
        if index_indicator == 0 and self.visual_word_generation:
            ind = ind & gts.ne(Constants.MASK)
        
        predict_res = preds.max(-1)[1][ind]
        target_res = gts[ind]

        self.word_acc_recorder[index_indicator].update(
                    (predict_res == target_res).sum().item(),
                    predict_res.size(0), 
                    multiply=False
            )

    def calculate_perplexity(self, index_indicator, preds, gts):
        # for the methods with visual word generation
        # we only compute the perplexity of the caption genration process
        if index_indicator == 0 and self.visual_word_generation:
            return None

        assert len(preds.shape) == 3
        assert preds.shape[:-1] == gts.shape

        log_probs = preds.gather(2, gts.unsqueeze(2)).squeeze(2)
        mask = gts.ne(Constants.PAD)
        num_words = float(torch.sum(mask))

        per_word_cross_entropy = -torch.sum(log_probs * mask) / num_words
        self.perplexity_recorder.update(per_word_cross_entropy.item(), num_words)

    def get_fieldsnames(self):
        return ['Word Acc%d' % i for i in range(self.num_word_acc)] + ['Perplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()


class Criterion(object):
    """
        Calculating losses or some metrics for all tasks

        Standard operations:
            1. before a epoch, Criterion.reset_loss_recorder()
            2. during a epoch, Criterion.get_loss(forward_results)
            3. after  a epoch, Criterion.get_loss_info()
    """ 
    def __init__(self, opt, crit_objects, keys, names, scales, summarywriter=None):
        assert len(crit_objects) == len(keys)
        assert len(keys) == len(names)
        assert len(names) == len(scales)
        self.opt = opt
        self.crit_objects = crit_objects
        self.num_loss = len(crit_objects)
        self.keys = keys
        self.names = names
        self.scales = scales
        self.summarywriter = summarywriter
        self.n_current_round = 0
        
    def reset_loss_recorder(self):
        self.loss_recorder = [AverageMeter() for _ in range(self.num_loss)]
        for crit_object in self.crit_objects:
            if getattr(crit_object, 'reset_recorder', None) is not None:
                crit_object.reset_recorder()

    def get_loss(self, results, **kwargs):
        """
            args:
                results: dict, contains the forward results of the model and some ground-truths
        """
        loss = []
        for i in range(self.num_loss):
            # calculate the i-th loss
            if isinstance(self.crit_objects[i], CritBase):
                # if 'new_module_prediction' in results.keys():
                #     new_module_loss, i_loss, num_samples = self.crit_objects[i](results)
                #     print('new_module_loss:', new_module_loss)
                # else:
                #     i_loss, num_samples = self.crit_objects[i](results)
                if self.opt['is_FCNet']:
                    FCNet_loss, i_loss, num_samples = self.crit_objects[i](results)
                elif self.opt['is_BCNet']:
                    # BCNet_loss1, BCNet_loss2, i_loss, num_samples = self.crit_objects[i](results)
                    BCNet_loss1, i_loss, num_samples = self.crit_objects[i](results)
                elif self.opt['is_NAR_decoder']:
                    NAR_loss, i_loss, num_samples = self.crit_objects[i](results)
                elif self.opt['is_verb_decoder']:
                    verb_loss, i_loss, num_samples = self.crit_objects[i](results)
                else:
                    i_loss, num_samples = self.crit_objects[i](results)
            else:
                # prepare the predictions and its corresponding ground-truths
                preds = results[self.keys[i][0]]
                gts = results[self.keys[i][1]]
                i_loss = self.crit_objects[i](preds, gts)
                num_samples = gts.size(0)
        
            # weighting the i-th loss
            loss.append(i_loss * self.scales[i])

            # update the statistics of the i-th loss
            self.loss_recorder[i].update(i_loss.item(), num_samples)

        # loss = loss1 * scale1 + loss2 * scale2 + ... 
        loss = torch.stack(loss, dim=0).sum(0)
        if self.opt['is_FCNet']:
            return FCNet_loss, loss
        elif self.opt['is_BCNet']:
            # return BCNet_loss1, BCNet_loss2, loss
            return BCNet_loss1, loss
        elif self.opt['is_NAR_decoder']:
            return NAR_loss, loss
        elif self.opt['is_verb_decoder']:
            return verb_loss, loss
        else:
            return loss


    def get_loss_info(self):
        all_names = self.names
        all_info = [meter.avg for meter in self.loss_recorder]

        for crit_object in self.crit_objects:
            if getattr(crit_object, 'get_info', None) is not None:
                this_name, this_info = crit_object.get_info()
                all_names += this_name
                all_info += this_info

        if self.summarywriter is not None:
            self.n_current_round += 1
            for name, loss in zip(all_names, all_info):
                self.summarywriter.add_scalar(name, loss, global_step=self.n_current_round)

        # e.g., ['Cap Loss', 'Word Acc0', 'Perplexity'], [31.8, 0.385, 53.0]
        return all_names, all_info
    
    def get_fieldsnames(self):
        exclude_index_set = []
        fieldsnames = []
        for i, crit_object in enumerate(self.crit_objects):
            if isinstance(crit_object, LanguageGeneration):
                exclude_index_set.append(i)
            elif getattr(crit_object, 'get_fieldsnames', None) is not None:
                fieldsnames += crit_object.get_fieldsnames()

        fieldsnames += [n for i, n in enumerate(self.names) if i not in exclude_index_set]                
        return fieldsnames


def get_criterion(opt, summarywriter=None):
    assert isinstance(opt['crit'], list)

    crit_objects = []
    for item in opt['crit']:
        crit_name = item.lower()
        if crit_name == 'lang':
            this_crit_object = LanguageGeneration(opt, crit_name)
        elif crit_name == 'length':
            this_crit_object = nn.KLDivLoss()
        else:
            raise NotImplementedError('''Please make sure that:\n
                1) the criterion name \'{}\' can be found in config.Constants.mapping.keys();\n
                2) the coressponding criterion for \'{}\' has been implemented in misc.crit;\n
                3) add \"elif crit_name == \'{}\': this_crit_object = xxx\" in misc.crit.get_criterion().\n
                '''.format(crit_name, crit_name, crit_name))

        crit_objects.append(this_crit_object)

    return Criterion(
            opt=opt,
            crit_objects=crit_objects,
            keys=opt['crit_key'],
            names=opt['crit_name'],
            scales=opt['crit_scale'],
            summarywriter=summarywriter
        )


def get_criterion_during_evaluation(opt, **kwargs):
    opt_for_crit = defaultdict(list)
    for key in ['attribute', 'length']:
        if key in opt['crit']:
            index_of_this_task = opt['crit'].index(key)
            for k in ['crit', 'crit_key', 'crit_name', 'crit_scale']:
                opt_for_crit[k].append(opt[k][index_of_this_task])
    if len(opt_for_crit):
        return get_criterion(opt_for_crit, **kwargs)
    return None
