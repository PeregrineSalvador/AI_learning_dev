import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset,DataLoader

import lightning as L

token_to_id = {
    'what' : 0,
    'is' : 1,
    'statquest' : 2,
    'awesome' : 3,
    '<EOS>' : 4,
}

id_to_token = dict(map(reversed, token_to_id.items()))

# we map the tokens to id numbers because thr Pytorch word embedding function that we will use , nn.Embedding(),only accepts numbers as input

inputs = torch.tensor([[token_to_id["what"],
                        token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]],
                       [token_to_id["statquest"],
                        token_to_id["is"],
                        token_to_id["what"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]]])

labels = torch.tensor([[token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"],
                        token_to_id["<EOS>"]],
                      [token_to_id["is"],
                       token_to_id["what"],
                       token_to_id["<EOS>"],
                       token_to_id["awesome"],
                       token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)



class PositionEncoding(nn.Module):
    # d_model: 词嵌入向量的维度（默认=2，为了方便演示。实际中通常为512等大数）word embedding
    # max_len: 预设要处理的最大序列长度（默认=6） tokens
    def __init__(self, d_model=2, max_len=6):
        super().__init__()

        # 初始化一个全零矩阵，用来存储所有位置的位置编码
        # 其形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 创建一个表示所有token位置索引的张量
        # torch.arange(0, max_len) 生成 [0, 1, 2, 3, 4, 5]
        # .float() 转换为浮点数
        # .unsqueeze(1) 增加一个维度，从 [6] 变为 [6, 1]，以便后续进行广播计算
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)

        # 创建一个包含所有需要应用正弦函数的维度索引的张量
        # torch.arange(0, d_model, 2) 生成 [0, 2, 4, ...]，直到小于 d_model
        # 步长为2，意味着我们只取所有偶数维度的索引（因为奇偶维度要交替使用sin/cos）
        # 例如：d_model=2 -> embedding_index = [0]
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        # 计算除数项 div_term，这是公式中的 10000^(2i/d_model) 的倒数
        # torch.tensor(10000.0) 创建分母
        # embedding_index / d_model 计算 2i/d_model
        # ** 是幂运算
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)

        # 为所有位置的所有偶数维度 (0, 2, 4, ...) 赋值正弦函数计算结果
        # pe[:, 0::2] 表示：取所有行(:)，从第0列开始，步长为2的列（即所有偶数列）
        # torch.sin(position * div_term) 计算正弦值
        pe[:, 0::2] = torch.sin(position * div_term)

        # 为所有位置的所有奇数维度 (1, 3, 5, ...) 赋值余弦函数计算结果
        # pe[:, 1::2] 表示：取所有行(:)，从第1列开始，步长为2的列（即所有奇数列）
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将计算好的位置编码矩阵 pe 注册为模块的缓冲区（buffer）
        # 缓冲区是模型的一部分，其参数不会被优化器更新，但会随模型一起保存和加载
        self.register_buffer('pe', pe)

    # 定义前向传播方法
    # word_embeddings: 输入的网络嵌入向量，其形状应为 [序列长度, d_model]
    def forward(self, word_embeddings):
        # 将词嵌入与位置编码切片相加，并返回结果
        return word_embeddings + self.pe[:word_embeddings.size(0) , :]

#定义了缩放点积注意力机制层

class Attention(nn.Module):
    def __init__(self, d_model=2):
        '''
        这里标准定义的单头注意力
        这里的线性层初始化已经做好了合适的默认初始化，比如Xavier均匀初始化
        在大多数情况下，你不需要手动调整这些默认初始化
        除非你遇到了特定的训练困难或正在进行非常深入的模型研究。
        '''
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # 定义softmax计算的维度
        self.row_div = 0
        self.col_div = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        # 线性变换得到Query、Key、Value
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        # 计算相似度（注意力分数）
        sims = torch.matmul(q, k.transpose(dim0 = self.row_div, dim1 = self.col_dim))  # 形状: (batch_size, seq_len_q, seq_len_k)

        # 缩放（这里未显式缩放，实际常见实现会除以sqrt(d_k)）
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        # 掩码处理（若提供掩码）
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask == 0, value=-1e9)

        # 计算注意力权重（softmax归一化）
        attention_weights = F.softmax(scaled_sims, dim=self.col_dim)  # 沿key的序列维度做softmax

        # 计算加权和（注意力得分）
        attention_scores = torch.matmul(attention_weights, v)  # 形状: (batch_size, seq_len_q, d_model)

        return attention_scores

class DecoderOnlyTransformer(L.LightningModule):
    '''
    继承自 PyTorch Lightning 的 Module
    它封装了训练循环、验证、测试等逻辑，让研究者更专注于模型本身。
    '''
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim = 0),token_ids.size(dim = 1))))
        mask = mask == 0
        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)
model_input = torch.tensor(
    [token_to_id["what"],
     token_to_id["is"],
     token_to_id["statquest"],
     token_to_id["EOS"]])
input_length = model_input.size(dim=0)

predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])

predicted_ids = predicted_id

max_length = 6

for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break

model_input = torch.cat((model_input, predicted_id))
predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
predicted_ids = torch.cat((predicted_ids, predicted_id))
