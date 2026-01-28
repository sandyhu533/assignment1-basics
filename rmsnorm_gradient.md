# RMSNorm 求导公式

## 前向传播

对于输入 `x` 形状为 `(..., d_model)`：

```
ms = mean(x², dim=-1, keepdim=True)  # 形状: (..., 1)
rms = (ms + eps)^(-1/2)               # 形状: (..., 1)
result = x * rms * g                  # 形状: (..., d_model)
```

其中：
- `eps` 是小的常数（如 1e-5）
- `g` 是可学习参数，形状为 `(d_model,)`
- `*` 表示逐元素乘法（broadcasting）

## 1. 参数梯度：∂result/∂g

```
result = x * rms * g
∂result/∂g = x * rms
```

这是逐元素的，所以：
```
∂result_i/∂g_j = x_i * rms  (如果 i == j)
                = 0          (如果 i != j)
```

实际上，由于 broadcasting，`g` 的梯度是：
```
∂L/∂g = sum(∂L/∂result * x * rms, dim=0)
```

## 2. 输入梯度：∂result/∂x

这是更复杂的，因为 `rms` 依赖于整个 `x`。

### 步骤 1：计算 ∂rms/∂ms

```
rms = (ms + eps)^(-1/2)
∂rms/∂ms = -1/2 * (ms + eps)^(-3/2)
         = -1/2 * rms³
```

### 步骤 2：计算 ∂ms/∂x

```
ms = mean(x²) = (1/n) * sum(x²)
```

对于每个元素 `x_i`：
```
∂ms/∂x_i = (1/n) * 2 * x_i
         = 2 * x_i / n
```

其中 `n = d_model`。

### 步骤 3：计算 ∂rms/∂x

使用链式法则：
```
∂rms/∂x_i = ∂rms/∂ms * ∂ms/∂x_i
          = -1/2 * rms³ * 2 * x_i / n
          = -rms³ * x_i / n
```

### 步骤 4：计算 ∂result/∂x

```
result = x * rms * g
```

对于每个元素 `result_i = x_i * rms * g_i`：

```
∂result_i/∂x_j = ∂(x_i * rms * g_i)/∂x_j
```

这需要分两种情况：

**情况 1：i == j**
```
∂result_i/∂x_i = ∂(x_i * rms * g_i)/∂x_i
               = rms * g_i + x_i * g_i * ∂rms/∂x_i
               = rms * g_i + x_i * g_i * (-rms³ * x_i / n)
               = rms * g_i - g_i * rms³ * x_i² / n
               = rms * g_i * (1 - rms² * x_i² / n)
```

**情况 2：i != j**
```
∂result_i/∂x_j = x_i * g_i * ∂rms/∂x_j
               = x_i * g_i * (-rms³ * x_j / n)
               = -g_i * rms³ * x_i * x_j / n
```

### 向量形式

将所有元素组合起来，可以得到矩阵形式：

```
∂result/∂x = rms * g * I - (g * rms³ / n) * x * x^T
```

其中：
- `I` 是单位矩阵
- `g` 是对角矩阵，对角线元素是 `g`
- `x * x^T` 是外积

### 实际计算（用于反向传播）

在反向传播中，我们通常有上游梯度 `∂L/∂result`，需要计算 `∂L/∂x`：

```
∂L/∂x = (∂result/∂x)^T * (∂L/∂result)
```

展开后：
```
∂L/∂x = rms * g * (∂L/∂result) - (g * rms³ / n) * x * (x^T * (∂L/∂result))
```

可以简化为：
```
∂L/∂x = rms * g * (∂L/∂result) - (g * rms³ / n) * x * sum(x * (∂L/∂result))
```

## 总结

### 参数梯度
```
∂L/∂g = sum((∂L/∂result) * x * rms, dim=0)
```

### 输入梯度
```
∂L/∂x = rms * g * (∂L/∂result) - (g * rms³ / n) * x * sum(x * (∂L/∂result))
```

其中 `n = d_model`。
