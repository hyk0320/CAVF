import pickle

pkl_file1 = open('./MSR_VTT/info_corpus.pkl', 'rb')

data1 = pickle.load(pkl_file1)

verb_ = {}

for index1, vid in enumerate(data1["pos_tags"]):
    for index2, caption_pos in enumerate(data1["pos_tags"][vid]):
        for index3, pos in enumerate(data1["pos_tags"][vid][index2]):
            if pos == 8 and data1["captions"][vid][index2][index3] not in verb_.keys():
                verb_[data1["captions"][vid][index2][index3]] = data1["info"]["itow"][data1["captions"][vid][index2][index3]]

print(verb_, len(verb_.keys()))

with open("./MSR_VTT/MSRVTT_verb.pkl", "wb") as fo:
    data1_dump = pickle.dump(verb_, fo)

