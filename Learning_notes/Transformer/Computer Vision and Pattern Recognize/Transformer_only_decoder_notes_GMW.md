# Transformer_only_decoder个人解读

笔者在学习完LLM的前身——Transformer_only_decoder后，讲讲自己对其简单的看法，供大家学习参考。

## 注意力就是一切

众所周知，Transformer的核心创新在于其**自注意力机制**，它实现了一种革命性的**动态权重分配**范式。机器学习（ML）乃至神经网络（NN）的核心使命，归根结底是**学习已知，预测未知**。为了更好地看待和利用‘已知’信息，就必须更智能地规划这些信息的权重。自注意力机制所实现的这种基于全局上下文的、动态的权重分配能力，完美地契合了这一思想指导，显然，这是技术发展的一个必然。

## Transformer_only_decoder架构详解

笔者参考开源资料在GIT的团队中上传了Transformer_only_decoder.py，感兴趣和希望更好地看后文的读者可以打开Transformer_only_decoder.py。

### 数据加载

实际上，后续对于特征提取的第一个函数word_embedding只能对于输入为明白的数字进行处理，而不可能理解复杂的汉字又或是字母，所以数据处理中词汇映射的字典是非常关键的，即：

```python
token_to_id = {
    'what' : 0,
    'is' : 1,
    'statquest' : 2,
    'awesome' : 3,
    '<EOS>' : 4,
}

id_to_token = dict(map(reversed, token_to_id.items()))
```

注：为了便于理解数据的处理，这里用极少的数据来做演示，实际上的工程量肯定是远远大于这里的。

自然语言生成本质就是预测，因为做的是预测，自然相同维度的label自然是下一个维度的imput，即：

```python
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
```

### 位置编码

从语言逻辑本身来看待，位置信息一定是限定后面的词生成的约束条件，但是由于Transformer的自注意力机制本身没有位置信息，需要显式地添加位置编码。

注入位置信息的位置编码实际上是很难实现的，因为考虑自然语言逻辑，我们的位置信息应该包含众多性质：唯一性，确定性以及能表示相对位置和最重要的泛化能力。

主流的位置编码实现有三种：绝对位置编码、相对位置编码、旋转位置编码。本文是基于绝对位置编码的正弦余弦编码实现的位置编码，感兴趣的读者可以自行学习了解另外两种编码实现，这里给出相应的链接：[位置编码——绝对位置编码，相对位置编码，旋转位置编码](https://blog.csdn.net/qq_45791939/article/details/146075127?ops_request_misc=%7B%22request%5Fid%22%3A%2227c1ec09e24ee2633cf4d4c75dd2516d%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=27c1ec09e24ee2633cf4d4c75dd2516d&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-146075127-null-null.142^v102^control&utm_term=位置编码&spm=1018.2226.3001.4187)

原始Transformer论文提出的正弦/余弦编码公式为：
$$
\begin{aligned}

PE_{(pos,2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right) \\

PE_{(pos,2i+1)} &= \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)

\end{aligned}
$$
正弦和余弦函数具有数学上的**和差化积**性质。这意味着，对于任意一个固定的位置偏移 *k*，位置 *p**os*+*k* 的编码都可以表示为位置 *p**os* 编码的一个**线性函数**，这种特性为模型提供了一个强大的**归纳偏置**（Inductive Bias）：模型无需费力学习所有可能的绝对位置组合，只需通过简单的线性变换，就能理解和利用 token 之间的**相对位置关系**。这是正弦/余弦编码最核心的优势之一。

三角函数天然的具有周期性，这种性质不仅给予了位置信息更稳定的输出，但是意味着面对高维信息，周期性也意味着重复概率大，虽然有着正余弦交替机制作为保底，但是不可否认，这种危险仍然存在。

不过我们这里只有很少的数据量，用正弦余弦编码实现位置编码，置信度相对很高了。

值得提醒的是：

```py
self.register_buffer('pe', pe)
```

缓冲区是模型的一部分，其参数不会被优化器更新，但会随模型一起保存和加载。

### 自注意力机制
缩放点积注意力是这里的核心实现。读者一定知道，注意力机制是动态权重分配的过程。实际上，缩放点积注意力（Scaled Dot-Product Attention）是一种通过计算查询（Query）与键（Key）的相似度，来为值（Value）分配权重，从而聚焦重要信息的机制。

点积的目的就是余弦范化，计算查询（Query）与键（Key）的相似度来确定权重。

其计算过程可以概括为以下公式：
$$
Attention(Q, K, V) = softmax(QK^T / √d_k) V
$$
**缩放因子（1/√d_k）** 是其中的关键设计。点积操作 `QK^T` 的结果在统计上其**方差会随着维度 `d_k` 的增大而增大**。这会导致在应用 softmax 函数时，某些位置的得分会远远高于 others，使得梯度变得非常小（进入饱和区），从而**不利于模型训练**

通过除以 `√d_k`，可以将点积结果的方差缩放回 1，**稳定训练过程**，并允许模型更有效地学习。 

#### 掩码处理

我们这里采用因果掩码来实现掩码处理，在序列生成任务中（如机器翻译、文本生成），模型需要根据**已经生成的部分**来预测下一个元素。这意味着在预测当前位置时，模型不应该知晓当前位置之后的任何未来信息。这种要求被称为**因果约束（Causal Constraint）** 或 **自回归（Autoregressive）** 性质。

其实，在这里提到自回归性质时，可能有读者会问，既然要逐步生成，为何不直接用循环神经网络（RNN）那样串行计算？而因果掩码的巧妙之处在于它通过掩码**在并行计算中模拟了串行的时序依赖关系**。

简单举例，读者就可以明白其中奥妙。

假设原始注意力分数为：

```python
[[2, 4, 1],
 [3, 1, 2],
 [5, 2, 3]]
```

应用掩码（将未来位置设为 `-inf`）后变为：

```python
[[2, -inf, -inf],
 [3,   1,  -inf],
 [5,   2,    3]]
```

经 Softmax 后，每行权重分布为：

```python
[[1.0, 0.0, 0.0],   # 位置0只关注自身
 [0.7, 0.3, 0.0],   # 位置1关注位置0和自身
 [0.5, 0.3, 0.2]]   # 位置2关注所有位置
```

（**用空间模拟时间，拥有时序效果的同时保留了并行运算能力，此处应该全体起立。**）

## Decoder-Only Transformer 模型

模型本身没有什么好聊的，调用所有上述组件后，加个fc_layer放大特征，配置Adam自主优化学习率的优化器和残差链接，来优化反向传播的梯度优化问题。

最后训练+预测就好。

笔者才疏学浅，反复看了很久才敢写出自己的一些拙见，这篇文章权作抛砖引玉，希望各位学长能够给我一些建议，使得这篇解读更加完善。



