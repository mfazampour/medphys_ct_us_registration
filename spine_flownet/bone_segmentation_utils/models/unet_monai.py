from monai.networks.nets import UNet, BasicUNet
from net_utils.utils import get_argparser_group
from typing import Sequence, Union


class UNetMonai(BasicUNet):
    def __init__(self, hparams):
        # initialize MONAI model
        super(UNetMonai, self).__init__(dimensions=hparams.dimensions, in_channels=hparams.in_channels,
                                        out_channels=hparams.out_channels, act=hparams.activation,
                                        dropout=hparams.dropout, upsample=hparams.upsample)
        # parameters
        self.hparams = hparams


    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--dimensions', type=int, default=2, help='UNet dimensions (default: 3)')
        specific_args.add_argument('--dropout', type=Union[float, tuple], default=0.0,
                                   help='dropout: dropout ratio (default: 0.0)')
        specific_args.add_argument('--activation', type=str, default='LeakyReLU', help='')
        specific_args.add_argument('--upsample', type=str, default='deconv',
                                   help='upsample: upsampling mode, available options are deconv, pixelshuffle, '
                                        'nontrainable (default: deconv)')
        return parser
