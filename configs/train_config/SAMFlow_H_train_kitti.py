from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'kitti'
_CN.suffix ='kitti'
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 3
_CN.sum_freq = 100
_CN.val_freq = 499999999
_CN.image_size = [400, 720]
# _CN.image_size = [370, 1226]
_CN.add_noise = True
_CN.critical_params = []


_CN.transformer = 'SAMFlow_H'

_CN.model = None 

_CN.FlowModel = CN()
_CN.FlowModel.sam_checkpoint = None
_CN.FlowModel.pe = 'linear'
_CN.FlowModel.dropout = 0.0
_CN.FlowModel.encoder_latent_dim = 256 # in twins, this is 256
_CN.FlowModel.query_latent_dim = 64
_CN.FlowModel.cost_latent_input_dim = 64
_CN.FlowModel.cost_latent_token_num = 8
_CN.FlowModel.cost_latent_dim = 128
_CN.FlowModel.arc_type = 'transformer'
_CN.FlowModel.cost_heads_num = 1
# encoder
_CN.FlowModel.pretrain = True
_CN.FlowModel.context_concat = False
_CN.FlowModel.encoder_depth = 3
_CN.FlowModel.feat_cross_attn = False
_CN.FlowModel.patch_size = 8
_CN.FlowModel.patch_embed = 'single'
_CN.FlowModel.no_pe = False
_CN.FlowModel.gma = "GMA"
_CN.FlowModel.kernel_size = 9
_CN.FlowModel.rm_res = True
_CN.FlowModel.vert_c_dim = 64
_CN.FlowModel.cost_encoder_res = True
_CN.FlowModel.cnet = 'twins'
_CN.FlowModel.fnet = 'twins'
_CN.FlowModel.no_sc = False
_CN.FlowModel.only_global = False
_CN.FlowModel.add_flow_token = True
_CN.FlowModel.use_mlp = False
_CN.FlowModel.vertical_conv = False

_CN.FlowModel.freeze_bn = True
_CN.FlowModel.ft_ver = False
_CN.FlowModel.sam_scale = 'H'
_CN.FlowModel.weighted_add = False

# decoder
_CN.FlowModel.decoder_depth = 12
_CN.FlowModel.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']


### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 50000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'

_CN.trainer.freeze_bn = False
_CN.trainer.accumulate_grad_batches = 2

def get_cfg():
    return _CN.clone()
