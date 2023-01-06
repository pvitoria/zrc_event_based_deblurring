import torch
import torch.nn as nn
from basicsr.archs.arch_util import  ResidualBlockNoBN, make_layer
from models.conv_lstm import ConvLSTM
from torchvision.ops import DeformConv2d



##############################################################################################################################
##################################################### Based on basicsr EDVR ##################################################
####################### https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/arch_util.py ########################
##############################################################################################################################


class DCN_sep(nn.Module):
    '''Use other features to generate offsets and masks'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=1):
        super(DCN_sep, self).__init__()
        self.kernel_size = kernel_size
        channels_ = deformable_groups * 3 * kernel_size * kernel_size
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=True)
        self.dcn_v2_conv = DeformConv2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=kernel_size // 2,
                                        dilation=dilation,
                                        groups=deformable_groups,
                                        bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            print('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = torch.sigmoid(mask)

        return self.dcn_v2_conv(input, offset, mask)
    
    

    
class EventPCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution (PCD). It estimates offsets from events.
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8, kernel_size=5, levels=3 ):
        super(EventPCDAlignment, self).__init__()
        
        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.levels = levels

        # Pyramids
        for i in range(self.levels, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvLSTM(num_feat, [num_feat], (3, 3), 1, True, True, False)
            if i == self.levels:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCN_sep(
                num_feat,
                num_feat,
                kernel_size,# 3
                 stride=1,
                padding=int(kernel_size/2),#1
                deformable_groups=deformable_groups)
            if i < self.levels:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, evs_feat_l, img_feat_l):
        """Align neighboring frame features to the reference frame features.
        Args:
            evs_feat_l (list[Tensor]): Events feature list. It contains three pyramid levels (L1, L2, L3), each with shape (b, t, c, h, w).
            img_feat_l (list[Tensor]): Image feature list. It contains three pyramid levels (L1, L2, L3), each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        aligned_feat = []
        for i in range(self.levels, 0, -1):
            level = f'l{i}'
            offset = self.offset_conv1[level](evs_feat_l[i - 1])[1][0][0]
            if i == self.levels:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))
            
            feat = self.dcn_pack[level](img_feat_l[i - 1], offset)

            if i < self.levels:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
            aligned_feat.insert(0, feat)
        return aligned_feat

    
    

    
class model(nn.Module):


    def __init__(self,
                 num_img_ch=3,
                 num_evs_ch=5,
                 num_feat=64,
                 deformable_groups=1,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 kernel_size=5,
                 activation=nn.Tanh()):
        super(model, self).__init__()
        # extract pyramid features for each image
        self.img_conv_first = nn.Conv2d(num_img_ch, num_feat, 3, 1, 1)
        self.img_feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.img_conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.img_conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.img_conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.img_conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # extract pyramid features for each event chunk
        self.evs_conv_first = nn.Conv2d(num_evs_ch, num_feat, 3, 1, 1)
        self.evs_feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.evs_conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.evs_conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.evs_conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.evs_conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # pcd module
        self.pcd_align = EventPCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups, kernel_size = kernel_size)
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # reconstruction
        self.reconstruction_l1 = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.reconstruction_l2 = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.reconstruction_l3 = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.conv_last_l1 = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.conv_last_l2 = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.conv_last_l3 = nn.Conv2d(num_feat, 3, 3, 1, 1)
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.activation = activation

    def forward(self, x1, x2):
        """
        Args:
            x1: Blurry image. It has shape (b, c1, h, w).
            x2: Event chunks: It contains n event chunks, and has shape (b, n, c2, h, w)
        Out:
            Sharp image. It has shape (b, c1, h, w)
        """
        b, c1, h, w = x1.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')
        
        # extract features for each frame
        # L1
        img_feat_l1 = self.lrelu(self.img_conv_first(x1))
        img_feat_l1 = self.img_feature_extraction(img_feat_l1)
        # L2
        img_feat_l2 = self.lrelu(self.img_conv_l2_1(img_feat_l1))
        img_feat_l2 = self.lrelu(self.img_conv_l2_2(img_feat_l2))
        # L3
        img_feat_l3 = self.lrelu(self.img_conv_l3_1(img_feat_l2))
        img_feat_l3 = self.lrelu(self.img_conv_l3_2(img_feat_l3))
        
        img_feat_l1 = img_feat_l1.view(b, -1, h, w)
        img_feat_l2 = img_feat_l2.view(b, -1, h // 2, w // 2)
        img_feat_l3 = img_feat_l3.view(b, -1, h // 4, w // 4)
        
        _, n, c2, _, _ = x2.size()
        
        # extract features for event chunk
        # L1
        evs_feat_l1 = self.lrelu(self.evs_conv_first(x2.view(-1, c2, h, w)))
        evs_feat_l1 = self.evs_feature_extraction(evs_feat_l1)
        # L2
        evs_feat_l2 = self.lrelu(self.evs_conv_l2_1(evs_feat_l1))
        evs_feat_l2 = self.lrelu(self.evs_conv_l2_2(evs_feat_l2))
        # L3
        evs_feat_l3 = self.lrelu(self.evs_conv_l3_1(evs_feat_l2))
        evs_feat_l3 = self.lrelu(self.evs_conv_l3_2(evs_feat_l3))

        evs_feat_l1 = evs_feat_l1.view(b, n, -1, h, w)
        evs_feat_l2 = evs_feat_l2.view(b, n, -1, h // 2, w // 2)
        evs_feat_l3 = evs_feat_l3.view(b, n, -1, h // 4, w // 4)
        
        # PCD alignment
        evs_feat_l = [  # events feature list
            evs_feat_l1, 
            evs_feat_l2,
            evs_feat_l3
        ]
        img_feat_l = [  # image feature list
            img_feat_l1, 
            img_feat_l2,
            img_feat_l3
        ]
        deblur_feat = self.pcd_align(evs_feat_l, img_feat_l)

        
        # l3 
        out_l3 = self.reconstruction_l3(deblur_feat[2])
        out_l3_conv = self.conv_last_l3(out_l3)
        out_im_l3 = self.activation(out_l3_conv)
        out_im_l3_out = self.upsample(self.upsample(out_im_l3))
        

        # l2
        out_l2 = self.reconstruction_l2(deblur_feat[1])
        upsample_out_l3 = self.upsample(out_l3)
        _,compress_l2 = self.compresion_l2([out_l2,upsample_out_l3])
        out_l2_conv = self.conv_last_l2(compress_l2)
        out_im_l2_out = self.upsample(self.activation(out_l2_conv)) 
        out_im_l2 = self.activation(out_l2_conv) + self.upsample(out_im_l3)
        
        # l1
        
        out_l1 = self.reconstruction_l1(deblur_feat[0])
        upsample_out_l2 = self.upsample(out_l2)
        _,compress_l1 = self.compresion_l1([out_l1,upsample_out_l2])
        out_l1_conv = self.conv_last_l1(compress_l1)
        out_im_l1_out = self.activation(out_l1_conv)  
        out_im_l1 = self.activation(out_l1_conv) + self.upsample(out_im_l2)

        return [out_im_l1, out_im_l2, out_im_l3]
    
    
