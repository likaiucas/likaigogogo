class DETR(nn.Module):
    """
    DETR的一个基本实现

    此演示Demo与论文中的完整DETR模型有一下不同点:
    * 使用的是可学习位置编码(论文中使用的是正余弦位置编码)
    * 位置编码在输入端传入 (原文在注意力模块传入)
    * 采用fc bbox 预测器 (没有采用MLP)
    该模型在 COCO val5k 上达到约 40 AP，在 Tesla V100 上以约 28 FPS 的速度运行。
    仅支持批量大小为1。
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要“随意”设置。
        super().__init__()

        # create ResNet-50 backbone
        # 创建Resnet50
        self.backbone = resnet50()
        # 删除最后的全连接层
        del self.backbone.fc

        # create conversion layer
        # 将骨干网络的输出特征图维度映射到Transformer输入所需的维度
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        # nheads代表多头注意力的"头数"
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # num_classes需要在原有来别数量上多加一个non-empty类
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # 标准DETR模型中最后的输出层由三个全连接层构成而非一个全连接层
        # bbox的形式是(x,y,w,h),因此是四维
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        # 用于解码器输入的位置编码,100代表最终解码出100个物体
        # 即对一张图片(最多)检测出100个物体
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        # 用于编码器输入的位置编码
        # 对特征图的行、列分别进行位置编码，而后会将两者结果拼接
        # 因此维度格式hidden_dim的一半,前128是x后128是y
        # nn.Parameter() 在指定Tensor中随机生成参数
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        # 将backbone的输出维度转换为Transformer输入所需的维度
        # h.shape = (1, hidden_dim, H, W)
        h = self.conv(x)


        # construct positional encodings
        H, W = h.shape[-2:]
        # Tensor.unsqueeze() 在指定位置插入新维度
        # Tensor.repeat() 沿某个维度复制Tensor
        # self.col_embed[:W].shape = (W, hidden_dim / 2) hidden_dim = 256
        # self.col_embed[:W].unsqueeze(0).shape = (1, W, 128)
        # self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1).shape = (H, W, 128)
        # torch.cat(...).flatten(0, 1).shape = (HxW, 256)
        # torch.cat(...).flatten(0, 1).unsqueeze(1).shape = (HxW, 256, 256)
        # pos.shape = (HxW, 1, 256) (HxW, 1, hidden_dim) 这里中间加一维是对应batch的维度
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        # 输出到Transformer中h的维度为(HxW, batch, hidden_dim),
        # query_pos的维度为(100, 1, hidden_dim)
        # Tensor.permute() 按照指定维度顺序对Tonser进行转职
        # h.flatten(2).shape = (1, hidden_dim, HxW)
        # h.flatten(2).permute(2, 0, 1).shape = (HxW, 1, hidden_dim)
        # h.shape = (1, 100, hidden_dim)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        # 输出预测物体类别(batch, 100, num_classes + 1)
        # 预测的物体bbox(batch, 100, 4)
        # 之所以sigmoid是因为回归的是归一化的值
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}
