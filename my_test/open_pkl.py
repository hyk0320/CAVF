import pickle

pkl_file1 = open('/my_test/MSR_VTT/info_corpus.pkl', 'rb')

data1 = pickle.load(pkl_file1)
#print(type(data1))

# data1的结构 {info : {}， caption : {}，Pos : {} }

# print(data1['info']['itop'])

# print(data1['info']['length_info'])

vid = input("视频id：")

tokens_gt = data1['captions']['video{}'.format(vid)]

ref = data1['info']['itow']



for index, tokens in enumerate(tokens_gt):
    sentence = []
    for token in tokens:
        for id in ref:
            if id == token:
                sentence.append(ref[id])
    print(sentence)



# data1.keys = ['info', 'captions', 'pos_tags']


# data1['info'].keys = ['split', 'vid2id', 'split_category', 'itoc', 'itow', 'itop', 'length_info']
#  tokens： data1['captions'] = ['video2960', 'video2636', 'video4311', 'video1844', 'video2213', 'video3513', 'video2218', 'video1974', 'video2755', 'video3597', 'video3830', 'video5249', 'video923', 'video6968', 'video2201', 'video106', 'video1835', 'video349', 'video1968', 'video6541', 'video3615', 'video1217', 'video4419', 'video177', 'video3932', 'video24', 'video5690', 'video6568', 'video5658', 'video1893', 'video3494', 'video2134', 'video1361', 'video2102', 'video3722', 'video2163', 'video5124', 'video6429', 'video294', 'video3687', 'video4601', 'video3518', 'video5985', 'video1919', 'video842', 'video1429', 'video4948', 'video4622', 'video3770', 'video507', 'video1701', 'video4976', 'video267', 'video4635', 'video1558',.........]
# data1['pos_tags'] = ['video2960', 'video2636', 'video4311', 'video1844', 'video2213', 'video3513', 'video2218', 'video1974', 'video2755', 'video3597', 'video3830', 'video5249', 'video923', 'video6968', 'video2201', 'video106', 'video1835', 'video349', 'video1968', 'video6541', 'video3615', 'video1217', 'video4419', 'video177', 'video3932', 'video24', 'video5690', 'video6568', 'video5658', 'video1893', 'video3494', 'video2134', 'video1361', 'video2102', 'video3722', 'video2163', 'video5124', 'video6429', 'video294', 'video3687', 'video4601', 'video3518', 'video5985', 'video1919', 'video842', 'video1429', 'video4948', 'video4622', 'video3770', 'video507', 'video1701', 'video4976', 'video267', 'video4635', 'video1558', 'video332', 'video2985',.........]

'''
    info：
        spilt : 数据集划分 ['train', 'validate', 'test']
        vid2vid ： null
        spilt_category : {train : [], test : []}   这个应该是视频所属类别
        itoc : id to category of video    MSRVTT一共是【0， 19】个视频类别{0: 9, 1: 16, 2: 9, 3: 8, 4: 14, 5: 13, 6: 13           }
        itow : id to word     {6: 'a', 7: 'is', 8: 'the', 9: 'in', 10: 'man'}
        itop : id to Part of Speech   词性与几个特殊词的token {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>', 4: '<mask>', 5: '<vis>', 6: 'DET', 7: 'NOUN', 8: 'VERB', 9: 'ADP', 10: 'CCONJ', 11: 'PRON', 12: 'ADV', 13: 'ADJ', 14: 'NUM', 15: 'PART', 16: 'X', 17: 'SYM', 18: 'INTJ'}
        length_info : 这个还真不是句子长度信息，后面再看是啥 {'video2960': [0, 0, 0, 0, 0, 0, 1, 6, 3, 1, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   }

'''

'''
    captions:
        {'video2960': [[2, 6, 39, 235, 307, 109, 23, 494, 1387, 9, 6, 20, 25, 3], [2, 6, 39, 114, 307, 68, 129, 12, 6, 20, 25, 3], [2, 6, 114, 7, 119, 9, 8, 361, 3], [2, 6, 21, 88, 6, 20, 25, 5113, 68, 494, 2172, 8, 58, 3], [2, 6, 21, 88, 752, 11, 1199, 1204, 3], [2, 6, 107, 31, 8, 494, 2172, 20, 25, 7, 49, 3], [2, 6, 20, 25, 114, 7, 243, 18, 9, 6, 1387, 3], [2, 6, 20, 25, 12, 6, 162, 281, 119, 109, 23, 494, 1681, 3], [2, 6, 20, 25, 12, 6, 131, 281, 3], [2, 6, 20, 93, 299, 31, 494, 2172, 3], [2, 6, 8936, 307, 68, 9, 23, 3305, 1387, 3], [2, 23, 281, 307, 109, 23, 3305, 1681, 9, 6, 20, 25, 3], [2, 1253, 11, 7820, 13, 23, 494, 2172, 20, 25, 3], [2, 299, 194, 12, 40, 22, 6, 25, 3], [2, 494, 2172, 20, 25, 1884, 114, 222, 68, 3], [2, 40, 7, 22, 6, 20, 25, 3], [2, 6, 8936, 307, 68, 9, 23, 3305, 1387, 3], [2, 6, 20, 25, 12, 6, 131, 281, 3], [2, 6, 39, 235, 307, 109, 23, 494, 1387, 9, 6, 20, 25, 3], [2, 6, 39, 114, 307, 68, 129, 12, 6, 20, 25, 3]]     }
        这个是某一个视频对应20条语句的token序列
'''

'''
    pos_tags:
        {'video2960': [[2, 6, 7, 7, 8, 9, 6, 7, 7, 9, 6, 7, 7, 3], [2, 6, 7, 7, 8, 9, 9, 9, 6, 7, 7, 3], [2, 6, 7, 8, 8, 9, 6, 7, 3], [2, 6, 7, 8, 6, 7, 7, 8, 9, 7, 7, 6, 7, 3], [2, 6, 7, 8, 7, 10, 7, 11, 3], [2, 6, 7, 9, 6, 7, 7, 7, 7, 8, 8, 3], [2, 6, 7, 7, 7, 8, 8, 12, 9, 6, 7, 3], [2, 6, 13, 7, 9, 6, 13, 13, 8, 9, 6, 7, 7, 3], [2, 6, 13, 7, 9, 6, 13, 7, 3], [2, 6, 7, 8, 7, 9, 7, 7, 3], [2, 6, 7, 8, 9, 9, 6, 7, 7, 3], [2, 6, 7, 8, 9, 6, 7, 7, 9, 6, 7, 7, 3], [2, 7, 10, 7, 9, 6, 7, 7, 7, 7, 3], [2, 7, 7, 9, 7, 8, 6, 7, 3], [2, 7, 7, 7, 7, 13, 7, 7, 9, 3], [2, 7, 8, 8, 6, 7, 7, 3], [2, 6, 7, 8, 9, 9, 6, 7, 7, 3], [2, 6, 13, 7, 9, 6, 13, 7, 3], [2, 6, 7, 7, 8, 9, 6, 7, 7, 9, 6, 7, 7, 3], [2, 6, 7, 7, 8, 9, 9, 9, 6, 7, 7, 3]]      }
        这是某一个视频对应20条语句的词性token序列
'''




#
# pkl_file2 = open('/Users/zhangyi/PycharmProjects/Non-Autoregressive-Video-Captioning-master/my_test/refs.pkl', 'rb')
#
# data2 = pickle.load(pkl_file2)
#
# # print(data2.keys())
#
# # data2 : {'video2960' : , 'video2636' : ,        }
#
# print(data2['video2636'])
#
# # data2['video2636'] = [{'image_id': 'video2636', 'cap_id': 0, 'caption': 'a man gets hit in the face with a chair during a wwf wrestling match'}, {'image_id': 'video2636', 'cap_id': 1, 'caption': 'a man has his head ran into a metal tray in a wrestling match'}, {'image_id': 'video2636', 'cap_id': 2, 'caption': 'a man hit another man with a chair in a wrestling ring and then is pinning him'}, {'image_id': 'video2636', 'cap_id': 3, 'caption': 'a man is hit with a trashcan lid'}, {'image_id': 'video2636', 'cap_id': 4, 'caption': 'a man is hitting another man with chair in wwf'}, {'image_id': 'video2636', 'cap_id': 5, 'caption': 'a man narrates a wf match between two men in a wrestling ring'}, {'image_id': 'video2636', 'cap_id': 6, 'caption': 'it is a wrestling show'}, {'image_id': 'video2636', 'cap_id': 7, 'caption': 'people are wrestling in a ring'}, {'image_id': 'video2636', 'cap_id': 8, 'caption': 'professional wrestlers battle in a wrestling ring'}, {'image_id': 'video2636', 'cap_id': 9, 'caption': 'this is a wwf wrestling match'}, {'image_id': 'video2636', 'cap_id': 10, 'caption': 'two man s are fighting with each other'}, {'image_id': 'video2636', 'cap_id': 11, 'caption': 'two men are wrestling each other with weapons'}, {'image_id': 'video2636', 'cap_id': 12, 'caption': 'two men wrestling and the umpire trying to judge'}, {'image_id': 'video2636', 'cap_id': 13, 'caption': 'two men wrestling in front of an audience'}, {'image_id': 'video2636', 'cap_id': 14, 'caption': 'wrestler throws other person against ring and hits him with a chair'}, {'image_id': 'video2636', 'cap_id': 15, 'caption': 'wrestlers are wrestling on a stage'}, {'image_id': 'video2636', 'cap_id': 16, 'caption': 'wrestling show on the ground'}, {'image_id': 'video2636', 'cap_id': 17, 'caption': 'wwe fight between hardy brothers'}, {'image_id': 'video2636', 'cap_id': 18, 'caption': 'people are wrestling in a ring'}, {'image_id': 'video2636', 'cap_id': 19, 'caption': 'a man gets hit in the face with a chair during a wwf wrestling match'}]
