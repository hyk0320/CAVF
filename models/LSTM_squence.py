
import torch.nn as nn
import torch

class LSTM_SQUENCE(nn.Module):
    def __init__(self, config):
        super(LSTM_SQUENCE, self).__init__()
        self.lstm = nn.LSTM(config.dim_hidden, config.dim_hidden, batch_first=True, dropout=0.5)
        self.batch_size = config.batch_size
        #self.linear = nn.Linear(config.dim_hidden, 1024) # 语料库里面是10547个ID对应这么多单词
        self.linear = nn.Linear(3*config.dim_hidden, config.dim_hidden)
        self.init_weights()

    def init_weights(self):
        self.lstm.weight_hh_l0.data.uniform_(-0.08, 0.08)
        self.lstm.weight_ih_l0.data.uniform_(-0.08, 0.08)
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)

        self.linear.weight.data.uniform_(-0.08, 0.08)
        self.linear.bias.data.fill_(0)

    def forward(self, pos_embeddings, tgt_seq, enc_output, device, hidden_states=None, iteration=0):
        # output1 = torch.Tensor()#   这个是用作输出到decoder
        # output2 = torch.Tensor()#   这个是用作计算交叉熵损失的,保存未映射（3*dim——1*dim）的结果

        self.lenOfTgt = tgt_seq.size(1)
        print(self.lenOfTgt)
        self.enc_output = enc_output
        self.pos_embeddings = pos_embeddings
        self.hidden_states = hidden_states
        self.iteration = iteration
        '''
            下面分为三个部分：
            一、第一个位置的输出，第二个位置的输出，第三个位置的输出
            二、concate以上是那个位置的输出结果
            三、线性映射：3*512-》512
        '''
        if self.lenOfTgt < 30:
            print('here is test begin!')
            print('iteration:', self.iteration)
            if self.iteration == 0:
                lstm_out1, (h1, c1) = self.lstm(self.enc_output + self.pos_embeddings)  # bs * len * 30

                EnAndPos1 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.linspace(1, self.lenOfTgt-1, self.lenOfTgt-1).long().to(device))  # bs * 29 * 512
                shape = list(EnAndPos1.size())
                PAD_word1 = torch.zeros([shape[0], 1, shape[2]]).to(device)
                tmp1 = torch.cat((EnAndPos1, PAD_word1), 1)

                lstm_out2, (h2, c2) = self.lstm(tmp1 + lstm_out1, (h1, c1))

                EnAndPos2 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.linspace(2, self.lenOfTgt-1, self.lenOfTgt-2).long().to(device))#bs * 28 *512
                shape = list(EnAndPos2.size())
                PAD_word2 = torch.zeros([shape[0], 2, shape[2]]).to(device)
                tmp2 = torch.cat((EnAndPos2, PAD_word2), 1)

                lstm_out3, _ = self.lstm(tmp2 + lstm_out2, (h2, c2))

                concat_three_lstm_result = torch.cat((lstm_out1, lstm_out2, lstm_out3), 2)#bs * len * (3*512)

                output1 = self.linear(concat_three_lstm_result)
                output1 = output1 + self.enc_output + self.pos_embeddings
                output2 = concat_three_lstm_result
                # print(type(output2))

            else:
                out1 = hidden_states

                h1 = torch.index_select(out1, 1, torch.linspace(1, self.lenOfTgt-1, self.lenOfTgt-1).long().to(device))
                shape = list(h1.size())
                PAD_word1 = torch.zeros([shape[0], 1, shape[2]]).to(device)
                tmp1 = torch.cat((h1, PAD_word1), 1)
                out2 = tmp1

                h2 = torch.index_select(out1, 1, torch.linspace(1, self.lenOfTgt-1, self.lenOfTgt-1).long().to(device))
                shape = list(h2.size())
                PAD_word1 = torch.zeros([shape[0], 1, shape[2]]).to(device)
                tmp2 = torch.cat((h2, PAD_word1), 1)
                out3 = tmp2

                concat_three_decoder_output = torch.cat((out1, out2, out3), 2)
                output1 = self.linear(concat_three_decoder_output)
                output1 = output1 + self.enc_output + self.pos_embeddings
                output2 = concat_three_decoder_output


            print('here is test end!')

            return output1, output2
        else:
            print('here is train begin!')
            '''
                这一块主要是将三次lstm的结果进行loss约束调整lstm
            '''
            lstm_out1, (h1, c1) = self.lstm(self.enc_output + self.pos_embeddings)# bs * 30 * 512


            EnAndPos1 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.linspace(1, self.lenOfTgt-1, self.lenOfTgt-1).long().to(device))#bs * 29 *512
            shape = list(EnAndPos1.size())# bs * 29 * 512
            PAD_word1 = torch.zeros([shape[0], 1, shape[2]]).to(device)
            tmp1 = torch.cat((EnAndPos1, PAD_word1), 1)

            #
            lstm_out2, (h2, c2) = self.lstm(tmp1 + lstm_out1, (h1, c1))


            EnAndPos2 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.linspace(2, self.lenOfTgt-1, self.lenOfTgt-2).long().to(device))#bs * 28 *512
            shape = list(EnAndPos2.size())# bs * 28 * 512
            PAD_word2 = torch.zeros([shape[0], 2, shape[2]]).to(device)
            tmp2 = torch.cat((EnAndPos2, PAD_word2), 1)

            #
            lstm_out3, _ = self.lstm(tmp2 + lstm_out2, (h2, c2))


            concat_three_lstm_result = torch.cat((lstm_out1, lstm_out2, lstm_out3), 2)  #bs * 30 * (3*512)

            output1 = self.linear(concat_three_lstm_result)
            output1 = output1 + self.enc_output + self.pos_embeddings
            output2 = concat_three_lstm_result
            # print(type(output2))

            """
                使用gt输入去训练decoder
            """
            Windows1 = self.hidden_states# tgt、pos

            print(self.hidden_states.shape)

            label_EnAndPos1 = torch.index_select(self.hidden_states, 1, torch.linspace(1, self.lenOfTgt-1, self.lenOfTgt-1).long().to(device))
            label_shape = list(label_EnAndPos1.size())
            label_PAD_word1 = torch.zeros([label_shape[0], 1, label_shape[2]]).to(device)
            Windows2 = torch.cat((label_EnAndPos1, label_PAD_word1), 1)

            label_EnAndPos2 = torch.index_select(self.hidden_states, 1, torch.linspace(2, self.lenOfTgt-1, self.lenOfTgt-2).long().to(device))#bs * 28 *512
            label_shape = list(label_EnAndPos2.size())
            label_PAD_word2 = torch.zeros([label_shape[0], 2, label_shape[2]]).to(device)
            Windows3 = torch.cat((label_EnAndPos2, label_PAD_word2), 1)

            label_output1 = self.linear(torch.cat((Windows1, Windows2, Windows3), 2))

            label_output1 = label_output1 + self.enc_output + self.pos_embeddings

            print('here is train end!')

            return label_output1, output2


        # for index in range(self.lenOfTgt - 2):
        #     # 经过三次LSTM  self.enc_output是512维，self.pos_embeddings[0][index]是512维，self.lstm输出也是512维
        #     print('in LSTM SQUENCE:', self.enc_output.shape, self.pos_embeddings.shape)
        #     EnAndPo1 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([index]).long().to(device)) # shape = bs * 1 * 512
        #     lstm_out1, _ = self.lstm(EnAndPo1)         # lstm_out.shape = (batch_size, seq_length=1, num_directions*hidden_size)  numdirections = 1
        #     reshape_lstm_out1 = torch.squeeze(lstm_out1, dim=1)# shape bs * 512
        #
        #     EnAndPo2 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([index+1]).long().to(device))
        #     lstm_out2, _ = self.lstm(EnAndPo2 + lstm_out1)
        #     reshape_lstm_out2 = torch.squeeze(lstm_out2, dim=1)
        #
        #     EnAndPo3 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([index+2]).long().to(device))
        #     lstm_out3, _ = self.lstm(EnAndPo3 + lstm_out2)
        #     reshape_lstm_out3 = torch.squeeze(lstm_out3, dim=1)
        #
        #     tmp1 = torch.cat((reshape_lstm_out1, reshape_lstm_out2, reshape_lstm_out3), 1) # 按列拼接 output[index].shape = (batch_size, 3*hidden_size)
        #     output1.append(self.linear(tmp1) ) #   output1.shape = batch_size * hidden_size
        #     output2.append(tmp1)
        # # 倒数第二个位置的索引
        # tmp = index + 1
        #
        # #   缺少的token进行<PAD>补足
        # shape = list(reshape_lstm_out3.size())
        # pad_word = torch.zeros(shape[0], shape[1]).to(device)
        # '''
        #     倒数第二个位置的结果
        # '''
        # EnAnd1 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([tmp]).long().to(device))
        # lstm_out1, _ = self.lstm(EnAnd1)
        # reshape_lstm_out1 = torch.squeeze(lstm_out1, dim=1)
        #
        # EnAndPo2 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([index+1]).long().to(device))
        # lstm_out2, _ = self.lstm(EnAndPo2 + lstm_out1)
        # reshape_lstm_out2 = torch.squeeze(lstm_out2, dim=1)
        #
        # tmp2 = torch.cat((reshape_lstm_out1, reshape_lstm_out2, pad_word), 1)
        # output1.append(self.linear(tmp2))
        # output2.append(tmp2)
        # '''
        #     最后一个位置的结果
        # '''
        # EnAnd1 = torch.index_select(self.enc_output + self.pos_embeddings, 1, torch.Tensor([tmp+1]).long().to(device))
        # lstm_out1, _ = self.lstm(EnAnd1)
        # reshape_lstm_out1 = torch.squeeze(lstm_out1, dim=1)
        #
        # tmp3 = torch.cat((reshape_lstm_out1, pad_word, pad_word), 1)
        # output1.append(self.linear(tmp3))
        # output2.append(tmp3)
        #
        # # output1是list = [tensor(1), tensor(2),   ],里面的tensor是bs * 512
        # new_output1 = torch.transpose(torch.stack(output1), 0, 1)# bs * max_len * 512
        # new_output2 = torch.transpose(torch.stack(output2), 0, 1)# bs * max_len * 3*512
        #
        # return new_output1, new_output2
        #
        #
