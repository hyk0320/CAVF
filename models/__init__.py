from .seq2seq import Seq2Seq
from models import Encoder
from models import Decoder

from models import FCNet_decoder

from models import FCNet_decoder_new

from models import BCNet_decoder

from models import visual_decoder

from models import NAR_decoder

from models import Verb_Predictor

from models import Predictor

from models import action_predictor

from .joint_representation import Joint_Representaion_Learner
import torch.nn as nn
from config import Constants


def print_info(module_name, supported_modules, key_name):
    print('Supported {}:'.format(key_name))
    for item in supported_modules:
        print('- {}{}'.format(item, '*' if module_name == item else ''))
    
    if module_name not in supported_modules:
        raise ValueError('We can not find {} in models/{}.py'.format(module_name, key_name))


def get_preEncoder(opt, input_size):
    assert not opt.get('use_preEncoder', False)
    return None, input_size.copy()


def get_encoder(opt, input_size):
    print_info(
        module_name=opt['encoder'],
        supported_modules=Encoder.__all__,
        key_name='Encoder'
    )
    return getattr(Encoder, opt['encoder'], None)(opt)


def get_joint_representation_learner(opt):
    if opt.get('no_joint_representation_learner', False):
        return None
    feats_size = [opt['dim_hidden']] * len(opt['modality'])
    return Joint_Representaion_Learner(feats_size, opt)


def get_auxiliary_task_predictor(opt):
    supported_auxiliary_tasks = [item[10:] for item in dir(Predictor) if 'Predictor_' in item]

    layers = []
    for crit_name in opt['crit']:
        if crit_name in supported_auxiliary_tasks:
            predictor_name = 'Predictor_%s'%crit_name
            _func = getattr(Predictor, predictor_name, None)
            if _func is None:
                raise ValueError('We can not find {} in models/Predictor.py'.format(predictor_name))
            layers.append(_func(opt, key_name=Constants.mapping[crit_name][0]))
    return None if not len(layers) else Predictor.Auxiliary_Task_Predictor(layers)

def get_action_predictor(opt):
    return getattr(action_predictor, "predict_action", None)(opt, "verb")


def get_decoder(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(Decoder, opt['decoder'], None)(opt)

def get_FCNet(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(FCNet_decoder, opt['decoder'], None)(opt)

def get_FCNet_new(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(FCNet_decoder_new, opt['decoder'], None)(opt)

def get_BCNet(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(BCNet_decoder, opt['decoder'], None)(opt)

def get_visual_decoder(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(visual_decoder, opt['decoder'], None)(opt)

def get_NAR_decoder(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(NAR_decoder, opt['decoder'], None)(opt)

def get_verb_decoder(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(Verb_Predictor, opt['decoder'], None)(opt)


def get_model(opt):
    modality = opt['modality'].lower()
    input_size = []
    mapping = {
        'i': opt['dim_i'],
        'm': opt['dim_m'],
        'a': opt['dim_a'],
        'o': opt['dim_o'],
    }
    for char in modality:
        assert char in mapping.keys()
        input_size.append(mapping[char])

    preEncoder, input_size = get_preEncoder(opt, input_size)
    encoder = get_encoder(opt, input_size)
    joint_representation_learner = get_joint_representation_learner(opt)
    have_auxiliary_tasks = sum([(1 if item not in ['lang'] else 0) for item in opt['crit']])
    auxiliary_task_predictor = get_auxiliary_task_predictor(opt)
    # if "is_verb_decoder" in opt.keys():
    #     v_action_predictor = get_action_predictor(opt)
    # else:
    #     v_action_predictor = None
    v_action_predictor = None
    if opt['is_FCNet']:
        # decoder = get_FCNet(opt)
        decoder = get_FCNet_new(opt)
    elif opt['is_BCNet']:
        decoder = get_BCNet(opt)
    elif opt['is_visual_decoder']:
        decoder = get_visual_decoder(opt)
    # elif opt['is_NAR_decoder']:
    #     decoder = get_NAR_decoder(opt)
    # else:
    #     decoder = get_decoder(opt)
    # elif opt["is_verb_decoder"]:
    #     decoder = get_verb_decoder(opt)
    else:
        # print("here")
        # print(opt['is_NAR_decoder'])
        # print(opt['num_attention_heads_c'])
        
        decoder = get_NAR_decoder(opt)
        # decoder = get_verb_decoder(opt)

    tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=False)

    model = Seq2Seq(
        opt=opt,
        preEncoder=preEncoder,
        encoder=encoder,
        joint_representation_learner=joint_representation_learner,
        auxiliary_task_predictor=auxiliary_task_predictor,
        action_predictor=v_action_predictor,
        decoder=decoder,
        tgt_word_prj=tgt_word_prj,
        )
    return model
