

## relative_position_bias_table和relative_position_index的理解
先上源代码：
```python
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # (2*Wh-1 * 2*Ww-1, nH)

coords_h = torch.arange(self.window_size[0])
coords_w = torch.arange(self.window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, Wh, Ww)
coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
relative_coords[:, :, 0] += self.window_size[0] - 1
relative_coords[:, :, 1] += self.window_size[1] - 1
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
self.register_buffer("relative_position_index", relative_position_index)
```
1. relative_position_bias_table本质。
   ```python
   # 将relative_position_bias_table注册为可训练参数
   self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # (2*Wh-1 * 2*Ww-1, nH)
   ```
   relative_position_bias_table其实与其他常见的bias是一致的，其特殊之处在于不直接使用，而是需要根据relative_position_index从其中获取特定的bias值，重新组成一个tensor，作为真正的bias，加到q@kT计算出的attn后。
2. relative_position_index的生成。  
   这要从attn=q@kT开始讲。首先，q, k的size为(B, C, pixs, code), kT的size为(B, C, code, pixs)， attn的size为(B, C, pixs, pixs)，pixs为像素数量，对于7x7的window，pixs=49，code是每个pix的编码长度，这种注意力mask产生的方法忽略了像素间本来存在的相对位置关系，导致无法充分利用特征中的信息。
   1. 为了体现出原来7x7窗口内像素的相对位置信息，首先获得每个点的坐标数组`coords_flatten`(2, 7*7)，2表示[x, y]两个轴向的坐标值。
   2. 然后通过对`coords_flatten`逐项相自减的方式获得相对坐标`relative_coords`(2, 49, 49)，2表示[x, y]两个轴向的相对坐标差值。(49, 49)表示`coords_flatten`中49组坐标分别与自身的49组坐标相减。
   3. 接下来需要将相对坐标`relative_coords`这个由[x, y]构成的数组映射到由int构成的索引数组上。首先，由于索引>=0, 需要通过
      ```python
      relative_coords[:, :, 0] += self.window_size[0] - 1
      relative_coords[:, :, 1] += self.window_size[1] - 1
      ```
      将数组坐标索引映射到自然数域内。然后，如果直接将`relative_coords`的两个方向的索引值相加或相乘，将难以区分类似于(2, 3)和(3, 2)这两种不同相对位置情况，解决方式是借鉴一位数组模拟二维数组的思路，给每一个x索引乘以y索引的长度(本例中即y索引最大值`2 * self.window_size[1] - 1`)，然后再加上y索引的值。这样就完成了映射。
3. 整体实现。  
   在前向过程中，模型根据相对位置索引`relative_position_index`, 从bias表中获取对应的bias，加到attn中，由于bias表可通过反向过程学习优化，因此模型就可以学习到attn的注意偏向。
