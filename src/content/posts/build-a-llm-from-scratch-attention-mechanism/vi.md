---
title: "Xây dựng một LLM từ đầu: Cơ chế Attention"
description: "Hướng dẫn dễ tiếp cận về các cơ chế attention, lấy cảm hứng từ Build a Large Language Model (From Scratch) của Sebastian Raschka."
published: 2024-11-18T20:00:00+08:00
updated: 2025-01-11T21:00:00+08:00
category: Engineering
tags:
  - ai
  - engineer
series: LLM from scratch
draft: false
lang: vi
translationKey: build-a-llm-from-scratch-attention-mechanism
---

Dù đã làm việc với LLM từ thời GPT-2, thực ra tôi chưa bao giờ tự mình xây dựng hoàn chỉnh một LLM từ đầu. Phần lớn kiến thức của tôi bắt nguồn từ sự tò mò, vì vậy tôi đã tự học rất nhiều mà không có bất kỳ cấu trúc hệ thống đúng nghĩa nào. Ngay cả trong khóa học ở đại học, chi tiết về cách xây dựng một LLM hoàn chỉnh cũng chỉ được đề cập sơ lược, mặc dù tôi đã hiểu khá sâu về cách tokenization hoạt động nhờ phải tự xây dựng một tokenizer. Trong loạt bài blog (hy vọng là vậy) này, tôi sẽ cố gắng chắt lọc những gì mình đã học với hy vọng chúng cũng hữu ích cho người khác trong tương lai!

> [!NOTE]
> 
> Disclaimer: This blog post is heavily inspired by the content of Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).
>
> Please give it a read if you really want to go through the process step by step and have an even more indepth look at it.


## 1. Input cơ bản

Trước khi tìm hiểu cách một attention block hoạt động, trước tiên hãy xác định input mà chúng ta sẽ làm việc cùng.

Lấy toy embedding sau làm ví dụ:

```python
# %% simplified attention
import torch

# These are the toy embedding of these words
inputs = torch.tensor(
  [
    [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]  # step     (x^6)
  ]
)
```

Đây là một tensor thể hiện word embedding ba chiều của 6 token trong input sau:

> Your journey starts with one step

Word embedding là biểu diễn dạng vector của dữ liệu trong một không gian liên tục, nhiều chiều, có khả năng nắm bắt các mối quan hệ ngữ nghĩa.

```python
[0.43, 0.15, 0.89] # this is the embedding of the token "Your" 
                   # with an embedding dimension of 3
```

Nếu thực hiện dot product giữa hai embedding, bạn sẽ nhận được một giá trị scalar biểu thị mức độ tương đồng giữa chúng. Đây là một khái niệm quan trọng giúp giải thích tại sao attention mechanism lại được xây dựng theo cách này.

Thông thường, embedding dimension sẽ lớn hơn nhiều (ví dụ GPT-3.5 có 12,288 dimension). Tuy nhiên, để minh họa, chúng ta thu nhỏ embedding xuống chỉ còn 3 dimension.

Trong bài viết này, chúng ta sẽ gọi toàn bộ tensor là input sequence, còn token đang được xử lý là query token. Lý do là LLM không tiếp nhận toàn bộ sequence cùng một lúc, mà lần lượt tiếp nhận từng token trong sequence để có thể tạo ra một output sequence mạch lạc theo trình tự.

## 2. Tổng quan về Attention Mechanism

Mục tiêu của attention mechanism là tính toán **context vector** của một query token.

Mục đích của **context vector** là tạo ra biểu diễn phong phú hơn cho từng token trong input sequence bằng cách kết hợp thông tin từ tất cả các token khác trong sequence. LLM sẽ dùng các trainable weight matrix để học cách xây dựng những context vector này sao cho chúng phù hợp để LLM tạo ra token tiếp theo.

Để có được context vector, trước tiên chúng ta cần tính **attention score**. Nó cho biết mức độ liên quan của từng token trong input sequence so với query token. Score càng cao thì token càng liên quan và query token càng nên dành nhiều "attention" cho token đó.

Attention score được tính bằng cách thực hiện dot product giữa embedding của query token và tất cả các token khác trong sequence (bao gồm chính nó).

```python
query = inputs[1]  # "journey" as the query token

attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)

print(attn_scores)
```

```text
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

Sau đó, chúng ta chuẩn hóa attention score này bằng hàm softmax để biến nó thành **attention weight**. Việc này đảm bảo tổng tất cả phần tử trong tensor bằng 1.

```python
attn_weights = torch.softmax(attn_scores, dim=0)

print("Attention weights:", attn_weights)
print("Sum:", attn_weights.sum())
```

```text
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
```

Lý do chúng ta dùng softmax là để xử lý tốt hơn các giá trị cực trị và tạo ra những đặc tính gradient thuận lợi hơn trong quá trình training.

Sau khi có attention weight, chúng ta có thể nhân nó với embedding của tất cả các token trong sequence rồi cộng lại để thu được context vector của query token.

```python
context_vec = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec += attn_weights[i]*x_i

print(context_vec) 
```

```text
tensor([0.4419, 0.6515, 0.5683])
```

Như có thể thấy, context vector có dimension giống hệt token embedding. Về cơ bản, nó chỉ là một phiên bản giàu thông tin hơn của embedding ban đầu.

## 3. Attention Block có thể training

### 3.1. Có gì mới?

Ví dụ trên là một phép tính đơn giản hóa nhằm minh họa khái niệm chung của attention mechanism. Có thể thấy rõ nó không có trainable weight nào, và đó là thứ chúng ta sẽ bổ sung trong phần này.

Để thực hiện điều tương tự như trên, chúng ta cần 3 trainable weight matrix mới có tên:

- query
- key
- value

Tương ứng với những gì đã mô tả ở trên. Với embedding của từng input token:

- nhân với query weight matrix, chúng ta thu được query vector. Query vector thay thế query embedding trong ví dụ đơn giản hóa.

- nhân với key weight matrix, chúng ta thu được key vector. Trong ví dụ đơn giản hóa, giá trị query vector và key vector của query token hoàn toàn giống nhau, nhưng ở đây chúng không nhất thiết phải giống nhau. Key value đặc biệt chỉ được dùng để tính attention score và attention weight. Theo phép so sánh, bạn có thể coi nó là một vector chỉ dùng để đo mức độ liên quan của token này với query token.

- nhân với value matrix, chúng ta thu được value vector. Trong ví dụ đơn giản hóa, token embedding chính là value vector. Trong phiên bản này, value vector là thứ bạn nhân với attention weight để thu được context vector. Theo phép so sánh, hãy coi value vector là biểu diễn nội dung thực sự của token sẽ được dùng để tính context vector.

Ý tưởng là những trainable matrix này sẽ thay đổi khi LLM học. Sau khi model xác định **keys** nào (một phần của sequence) liên quan nhất đến **query** (đối tượng trọng tâm), nó sẽ truy xuất các **values** tương ứng (giá trị số của token).

So với một database system, chúng ta có thể hình dung:

- query là search input (ví dụ: yellow)
- key là phần indexing của item (ví dụ: yellow, gold, ... )
- value là giá trị thực tế của item (ví dụ: #FFFF00, #FFD700)

### 3.2. Ý tưởng cơ bản

Để có được các query, key và value vector này (từ đây gọi chung là qkv), chúng ta lần lượt training các qkv weight matrix. Nó sẽ trông như sau:

```python 
x_2 = inputs[1] # the query is "journey"
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = d_in 

# Initialise the W_query, W_key, W_value weight matrix with random value
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

print("Query Trainable Weigth", W_query)
print("Key Trainable Weight", W_query)
print("Value Trainable Weigth", W_query)
```

Output như sau:

```text
Query Trainable Weigth Parameter containing:
tensor([[0.2961, 0.5166, 0.2517],
        [0.6886, 0.0740, 0.8665],
        [0.1366, 0.1025, 0.1841]], requires_grad=True)
Key Trainable Weight Parameter containing:
tensor([[0.2961, 0.5166, 0.2517],
        [0.6886, 0.0740, 0.8665],
        [0.1366, 0.1025, 0.1841]], requires_grad=True)
Value Trainable Weigth Parameter containing:
tensor([[0.2961, 0.5166, 0.2517],
        [0.6886, 0.0740, 0.8665],
        [0.1366, 0.1025, 0.1841]], requires_grad=True)
```

Trong thực tế, chúng ta muốn nhận được qkv vector có dimension giống input embedding dimension, vì vậy trong trường hợp này, ta mong đợi ba weight matrix 3 x 3 tương ứng cho qkv.

Hãy cùng đi qua quy trình tính các qkv vector cần thiết:

```python
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("Query vector\n", query_2)
print("Key vector\n", key_2)
print("Value vector\n", value_2)
```

```text
Query vector
 tensor([0.8520, 0.4161, 1.0138], grad_fn=<SqueezeBackward4>)
Key vector
 tensor([0.7305, 0.4227, 1.1993], grad_fn=<SqueezeBackward4>)
Value vector
 tensor([0.9074, 1.3518, 1.5075], grad_fn=<SqueezeBackward4>)
```

Với những vector này, chúng ta chỉ cần lặp lại bước ở phần 2 để lấy tất cả context vector tương ứng với query "journey".

Với cách hiểu đó, chúng ta nhận ra thực ra có thể thay module `nn.Parameter` bằng `nn.Linear`, vì một phép toán linear không có bias về cơ bản tương đương với matrix multiplication. Bên cạnh đó, `nn.Linear` cũng khởi tạo một matrix ổn định hơn nhiều cho mục đích tính toán. Vì vậy, chúng ta có Single Self-Attention Mechanism sau.

```python
class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
```

Trong method forward, chúng ta tạo tất cả qkv vector cho từng token, rồi dùng chúng để tính hiệu quả tất cả context vector của từng token cùng một lúc.

### 3.3. Hãy tập trung vào hiện tại

Hiện giờ, vấn đề là query token đang chú ý đến **ALL** token trong input sequence. Điều này không lý tưởng, vì chúng ta không muốn model chú ý đến token chưa xuất hiện; về trực giác, việc KHÔNG chú ý đến những điều xảy ra trong tương lai là hợp lý.

Chúng ta đưa vào causal / masked attention, trong đó tất cả future value được mask bằng giá trị 0 để chúng không còn tham gia vào phép tính context vector.

```python
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer is used so that we don't have to manually describe what device (CPU / GPU)
        # to use for calculation and will automatically be decided base on usage for us
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        print("keys\n", keys)
        print("queries\n", queries)
        print("values\n", values)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose

        print("attention scores\n", attn_scores)

        attn_scores.masked_fill_( self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  
        # `:num_tokens` to account for cases where the number of tokens in the batch 
        # is smaller than the supported context_size

        print("masked attention scores\n", attn_scores)

        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 )

        print("attention weights\n", attn_scores)

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

Nếu chạy CausalAttention ở trên:

```python
torch.manual_seed(123)

# This is a batch of 2 inputs with 6 tokens of embedding 3 in each inputs
batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.2)

context_vecs = ca(batch)

print("context vectors\n", context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

Chúng ta sẽ nhận được:

```text
keys
 tensor([[[ 0.2727, -0.4519,  0.2216],
         [ 0.1008, -0.7142, -0.1961],
         [ 0.1060, -0.7127, -0.1971],
         [ 0.0051, -0.3809, -0.1557],
         [ 0.1696, -0.4861, -0.1597],
         [-0.0388, -0.4213, -0.1501]],
        [[ 0.2727, -0.4519,  0.2216],
         [ 0.1008, -0.7142, -0.1961],
         [ 0.1060, -0.7127, -0.1971],
         [ 0.0051, -0.3809, -0.1557],
         [ 0.1696, -0.4861, -0.1597],
         [-0.0388, -0.4213, -0.1501]]], grad_fn=<UnsafeViewBackward0>)
queries
 tensor([[[-0.3536,  0.3965, -0.5740],
         [-0.3021, -0.0289, -0.8709],
         [-0.3015, -0.0232, -0.8628],
         [-0.1353, -0.0978, -0.4789],
         [-0.2052,  0.0870, -0.4744],
         [-0.1542, -0.1499, -0.5888]],
        [[-0.3536,  0.3965, -0.5740],
         [-0.3021, -0.0289, -0.8709],
         [-0.3015, -0.0232, -0.8628],
         [-0.1353, -0.0978, -0.4789],
         [-0.2052,  0.0870, -0.4744],
         [-0.1542, -0.1499, -0.5888]]], grad_fn=<UnsafeViewBackward0>)
values
 tensor([[[ 0.3326,  0.5659, -0.3132],
         [ 0.3558,  0.5643, -0.1536],
         [ 0.3412,  0.5522, -0.1574],
         [ 0.2123,  0.2991, -0.0360],
         [-0.0177,  0.1780, -0.1805],
         [ 0.3660,  0.4382, -0.0080]],
        [[ 0.3326,  0.5659, -0.3132],
         [ 0.3558,  0.5643, -0.1536],
         [ 0.3412,  0.5522, -0.1574],
         [ 0.2123,  0.2991, -0.0360],
         [-0.0177,  0.1780, -0.1805],
         [ 0.3660,  0.4382, -0.0080]]], grad_fn=<UnsafeViewBackward0>)
attention scores
 tensor([[[-0.4028, -0.2063, -0.2069, -0.0635, -0.1611, -0.0672],
         [-0.2623,  0.1610,  0.1602,  0.1450,  0.1019,  0.1546],
         [-0.2630,  0.1553,  0.1546,  0.1416,  0.0979,  0.1510],
         [-0.0989,  0.1501,  0.1497,  0.1111,  0.1010,  0.1183],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,  0.0425],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]],
        [[-0.4028, -0.2063, -0.2069, -0.0635, -0.1611, -0.0672],
         [-0.2623,  0.1610,  0.1602,  0.1450,  0.1019,  0.1546],
         [-0.2630,  0.1553,  0.1546,  0.1416,  0.0979,  0.1510],
         [-0.0989,  0.1501,  0.1497,  0.1111,  0.1010,  0.1183],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,  0.0425],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]]],
       grad_fn=<UnsafeViewBackward0>)
masked attention scores
 tensor([[[-0.4028,    -inf,    -inf,    -inf,    -inf,    -inf],
         [-0.2623,  0.1610,    -inf,    -inf,    -inf,    -inf],
         [-0.2630,  0.1553,  0.1546,    -inf,    -inf,    -inf],
         [-0.0989,  0.1501,  0.1497,  0.1111,    -inf,    -inf],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,    -inf],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]],
        [[-0.4028,    -inf,    -inf,    -inf,    -inf,    -inf],
         [-0.2623,  0.1610,    -inf,    -inf,    -inf,    -inf],
         [-0.2630,  0.1553,  0.1546,    -inf,    -inf,    -inf],
         [-0.0989,  0.1501,  0.1497,  0.1111,    -inf,    -inf],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,    -inf],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]]],
       grad_fn=<MaskedFillBackward0>)
attention weights
 tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4392, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2820, 0.3591, 0.3589, 0.0000, 0.0000, 0.0000],
         [0.2253, 0.2602, 0.2601, 0.2544, 0.0000, 0.0000],
         [0.1809, 0.2043, 0.2042, 0.2078, 0.2029, 0.0000],
         [0.1456, 0.1743, 0.1743, 0.1685, 0.1678, 0.1694]],
        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4392, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2820, 0.3591, 0.3589, 0.0000, 0.0000, 0.0000],
         [0.2253, 0.2602, 0.2601, 0.2544, 0.0000, 0.0000],
         [0.1809, 0.2043, 0.2042, 0.2078, 0.2029, 0.0000],
         [0.1456, 0.1743, 0.1743, 0.1685, 0.1678, 0.1694]]],
       grad_fn=<SoftmaxBackward0>)
dropout attention weights
 tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.5490, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.4488, 0.4486, 0.0000, 0.0000, 0.0000],
         [0.2817, 0.3252, 0.3251, 0.0000, 0.0000, 0.0000],
         [0.2261, 0.2553, 0.0000, 0.2597, 0.2536, 0.0000],
         [0.1820, 0.2179, 0.2179, 0.2106, 0.0000, 0.2118]],
        [[1.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.7010, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.3525, 0.4488, 0.4486, 0.0000, 0.0000, 0.0000],
         [0.2817, 0.3252, 0.3251, 0.3180, 0.0000, 0.0000],
         [0.2261, 0.0000, 0.2553, 0.2597, 0.2536, 0.0000],
         [0.1820, 0.0000, 0.2179, 0.2106, 0.2098, 0.0000]]],
       grad_fn=<MulBackward0>)
context vectors
 tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1826,  0.3107, -0.1719],
         [ 0.3128,  0.5010, -0.1396],
         [ 0.3203,  0.5225, -0.1893],
         [ 0.2167,  0.3949, -0.1651],
         [ 0.3346,  0.5021, -0.1340]],
        [[ 0.4158,  0.7074, -0.3914],
         [ 0.2494,  0.3956, -0.1077],
         [ 0.4300,  0.7005, -0.2500],
         [ 0.3878,  0.6176, -0.2008],
         [ 0.2129,  0.3917, -0.1661],
         [ 0.1759,  0.3237, -0.1367]]], grad_fn=<UnsafeViewBackward0>)
context_vecs.shape: torch.Size([2, 6, 3])
```

Hãy phân tích cách class CausalAttention hoạt động.

Đầu tiên, trong phần initialization, chúng ta thêm 2 thành phần mới:
- `Dropout`
- `register_buffer`

Nói một cách chặt chẽ, bạn không cần `register_buffer` để class hoạt động, nhưng nó giúp đơn giản hóa quá trình gán phép tính cho device phù hợp. `Dropout` là một module cho phép bạn ngẫu nhiên đặt một số phần tử của input tensor về 0 (hay drop chúng) với một xác suất nhất định. Vì vậy, nếu dropout rate là 0.2, thì với tensor có 10 phần tử, 2 phần tử sẽ bị drop ngẫu nhiên.

Tiếp theo, chúng ta xem method forward. Input x thực ra có dạng batch data, trong đó:

> x.shape = [num_batch,num_tokens, d_in]

Sau đó, input x được đưa qua module `Linear` để lấy các qkv weight matrix tương ứng. Điều quan trọng cần lưu ý là trong Pytorch, một `nn.Module` có đặc điểm sau sau khi initialization:

```python
module = nn.Module(foo,bar,baz)

module(x)
# is the same as
module.forward(x)
```

Nói cách khác:

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

self.W_query(x)
# is the same as
self.W_query().forward(x)
# in this case it is carrying out the matrix multiplication
# between x and the arbitrary initial matrix

ca = CausalAttention(d_in, d_out, context_length, 0.0)

ca(inputs)
# is the same as
ca.forward(inputs)
# This is possible because the CausalAttention class inherit from the nn.Module class
```

Đây là một pattern rất phổ biến trong Pytorch nhưng thường bị những người giàu kinh nghiệm lướt qua khi giải thích các cách làm thông dụng của Pytorch. Vì vậy, để tránh nhầm lẫn, chỉ cần hiểu rằng nếu một `nn.Module` đã được initialized, bạn có thể gọi class như một function và ngầm hiểu rằng nó đang gọi method `forward`.

Shape nhận được từ `keys`, `queries`, `values` bằng `[num_batch,num_tokens, d_in]`. Chúng ta có thể thấy điều này từ output:

```text
keys
 tensor([[[ 0.2727, -0.4519,  0.2216],
         [ 0.1008, -0.7142, -0.1961],
         [ 0.1060, -0.7127, -0.1971],
         [ 0.0051, -0.3809, -0.1557],
         [ 0.1696, -0.4861, -0.1597],
         [-0.0388, -0.4213, -0.1501]],
        [[ 0.2727, -0.4519,  0.2216],
         [ 0.1008, -0.7142, -0.1961],
         [ 0.1060, -0.7127, -0.1971],
         [ 0.0051, -0.3809, -0.1557],
         [ 0.1696, -0.4861, -0.1597],
         [-0.0388, -0.4213, -0.1501]]], grad_fn=<UnsafeViewBackward0>)
```

Lý do cả hai batch có giá trị hoàn toàn giống nhau là vì input chúng ta cung cấp gồm 2 batch có cùng một input. Nhưng như có thể thấy rõ, có hai matrix 6 x 3, mỗi matrix lần lượt thuộc về một batch.

Sau đó, chúng ta tính attention score. Có thể thấy keys được transpose trước khi thực hiện matrix multiplication với queries. `keys.transpose(1,2)` sẽ trông như sau:

```text
tensor([[[ 0.2727,  0.1008,  0.1060,  0.0051,  0.1696, -0.0388],
         [-0.4519, -0.7142, -0.7127, -0.3809, -0.4861, -0.4213],
         [ 0.2216, -0.1961, -0.1971, -0.1557, -0.1597, -0.1501]],
        [[ 0.2727,  0.1008,  0.1060,  0.0051,  0.1696, -0.0388],
         [-0.4519, -0.7142, -0.7127, -0.3809, -0.4861, -0.4213],
         [ 0.2216, -0.1961, -0.1971, -0.1557, -0.1597, -0.1501]]])
```

Về cơ bản, chúng ta chỉ transpose keys vector theo dimension thứ 2 và thứ 3 (không phải thứ 1 và thứ 2 vì index trong Python bắt đầu từ 0). Vì vậy, nếu shape của `keys` là `[2, 6, 3]` thì `keys.transpose(1,2)` là `[2, 3, 6]`. Điều này giống với những gì đã trình bày trước đó khi thực hiện multiplication giữa query và key trong `SelfAttention`, chỉ là lặp lại với thêm một batch data.

Tiếp theo, chúng ta mask attention score. Như bạn có thể thấy, mọi thứ phía trên đường chéo là `-inf`, hay âm vô cực. Lý do là chúng ta muốn chuẩn hóa attention score để thu được attention weight, nhưng nếu chỉ đặt bằng 0 thì các phần tử phía trên đường chéo vẫn có giá trị weight do cách `softmax` hoạt động. Chúng ta không muốn điều đó. Để thực sự đưa weight về 0, ta cần dùng `-inf`.

```text
masked attention scores
 tensor([[[-0.4028,    -inf,    -inf,    -inf,    -inf,    -inf],
         [-0.2623,  0.1610,    -inf,    -inf,    -inf,    -inf],
         [-0.2630,  0.1553,  0.1546,    -inf,    -inf,    -inf],
         [-0.0989,  0.1501,  0.1497,  0.1111,    -inf,    -inf],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,    -inf],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]],
        [[-0.4028,    -inf,    -inf,    -inf,    -inf,    -inf],
         [-0.2623,  0.1610,    -inf,    -inf,    -inf,    -inf],
         [-0.2630,  0.1553,  0.1546,    -inf,    -inf,    -inf],
         [-0.0989,  0.1501,  0.1497,  0.1111,    -inf,    -inf],
         [-0.2004,  0.0102,  0.0098,  0.0397, -0.0013,    -inf],
         [-0.1048,  0.2070,  0.2065,  0.1480,  0.1407,  0.1575]]],
       grad_fn=<MaskedFillBackward0>)
```

Kết quả là các attention weight sau:

```text
attention weights
 tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4392, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2820, 0.3591, 0.3589, 0.0000, 0.0000, 0.0000],
         [0.2253, 0.2602, 0.2601, 0.2544, 0.0000, 0.0000],
         [0.1809, 0.2043, 0.2042, 0.2078, 0.2029, 0.0000],
         [0.1456, 0.1743, 0.1743, 0.1685, 0.1678, 0.1694]],
        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4392, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2820, 0.3591, 0.3589, 0.0000, 0.0000, 0.0000],
         [0.2253, 0.2602, 0.2601, 0.2544, 0.0000, 0.0000],
         [0.1809, 0.2043, 0.2042, 0.2078, 0.2029, 0.0000],
         [0.1456, 0.1743, 0.1743, 0.1685, 0.1678, 0.1694]]],
       grad_fn=<SoftmaxBackward0>)```

Tiếp theo, chúng ta áp dụng dropout rate 0.2 (hay 20%) lên attention weight để nhận được kết quả sau:

```text
dropout attention weights
 tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.5490, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.4488, 0.4486, 0.0000, 0.0000, 0.0000],
         [0.2817, 0.3252, 0.3251, 0.0000, 0.0000, 0.0000],
         [0.2261, 0.2553, 0.0000, 0.2597, 0.2536, 0.0000],
         [0.1820, 0.2179, 0.2179, 0.2106, 0.0000, 0.2118]],
        [[1.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.7010, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.3525, 0.4488, 0.4486, 0.0000, 0.0000, 0.0000],
         [0.2817, 0.3252, 0.3251, 0.3180, 0.0000, 0.0000],
         [0.2261, 0.0000, 0.2553, 0.2597, 0.2536, 0.0000],
         [0.1820, 0.0000, 0.2179, 0.2106, 0.2098, 0.0000]]],
       grad_fn=<MulBackward0>)
```

Có thể thấy khi dropout được áp dụng, nó sẽ đưa một số giá trị về 0, đồng thời giúp tính lại tất cả attention weight để đảm bảo các weight còn lại "heavier" nhằm bù cho những weight đã bị đưa về 0.

Cuối cùng, chúng ta tính context vector.

### 3.4. Tại sao chỉ dừng ở một head?

Trong thực tế, chúng ta dùng Multi-Head Attention thay vì chỉ một Causal Attention. Có thể hình dung Multi-Head attention như một nhóm Single Causal Attention được tính song song rồi kết hợp lại ở cuối.

Hãy thử sửa đổi Causal Attention ở trên để tạo ra một kiến trúc Multi-Head attention.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        print("keys", keys)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        print("keys\n", keys.shape)

        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        print("keys transpose\n", keys.shape)

        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        print("attn_score keys transpose\n", keys.transpose(2,3).shape)
        print("attn_scores\n", attn_scores.shape)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        print("mask_bool:\n", mask_bool)

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        print("attn_weights\n", attn_scores.shape)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        print("attn_weights\n", attn_scores.shape)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        print("context_vec shape:\n", context_vec.shape)
        print("context_vec:\n", context_vec)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
```

Để đảm bảo tính nhất quán, chúng ta sẽ dùng configuration sau làm input:

```python
torch.manual_seed(123)

batch = torch.stack((inputs, ), dim=0)

batch_size, context_length, d_in = batch.shape
d_out = 3
mha = MultiHeadAttention(d_in, d_out, context_length, 0, num_heads=3)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

Chúng ta có dropout rate bằng 0 và 3 attention head. d_out phải chia hết cho số head, vì nếu không, chúng ta sẽ không thể chia d_out đối xứng cho từng head. Điều này sẽ rõ ràng hơn khi xem code sau đây. Chúng ta cũng giảm batch xuống chỉ còn 1 input để dễ trình bày kết quả và thực hiện phép tính minh họa.

Chúng ta cũng in shape của keys vì hiển nhiên queries và values có cùng shape, do chúng có cùng phép transpose. Bây giờ hãy xem xét đến phần tính `attn_scores` trong method `forward`.

```text
keys tensor([[[ 0.2727, -0.4519,  0.2216],
         [ 0.1008, -0.7142, -0.1961],
         [ 0.1060, -0.7127, -0.1971],
         [ 0.0051, -0.3809, -0.1557],
         [ 0.1696, -0.4861, -0.1597],
         [-0.0388, -0.4213, -0.1501]]], grad_fn=<UnsafeViewBackward0>)
keys
 torch.Size([1, 6, 3, 1])
keys transpose
 torch.Size([1, 3, 6, 1])
attn_score keys transpose
 torch.Size([1, 3, 1, 6])
attn_scores
 torch.Size([1, 3, 6, 6])
```

Như có thể thấy ngay, chúng ta vẫn có các attn_score matrix 6x6 cho từng attention head trong số 3 head. Điểm khác biệt ở đây là mỗi head nắm bắt 1/3 lượng thông tin so với kiến trúc Causal Attention. Vì mỗi head chỉ nắm bắt một phần input sequence, chúng chỉ attend đến các subset khác nhau của sequence, nhưng khi kết hợp lại, chúng nắm bắt được pattern đa dạng hơn nhiều so với thiết kế Single / Causal Attention.

Phần tiếp theo thực hiện cùng phép tính đã trình bày trước đó. Chúng chỉ cần được transpose để data được sắp xếp theo đúng dimension cho matrix multiplication.

```text
mask_bool:
 tensor([[False,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True],
        [False, False, False, False,  True,  True],
        [False, False, False, False, False,  True],
        [False, False, False, False, False, False]])
attn_weights
 torch.Size([1, 3, 6, 6])
context_vec shape:
 torch.Size([1, 6, 3, 1])
context_vec:
 tensor([[[[ 0.3326],
          [ 0.5659],
          [-0.3132]],
         [[ 0.3445],
          [ 0.5651],
          [-0.2191]],
         [[ 0.3434],
          [ 0.5608],
          [-0.1963]],
         [[ 0.3100],
          [ 0.4965],
          [-0.1586]],
         [[ 0.2448],
          [ 0.4308],
          [-0.1632]],
         [[ 0.2655],
          [ 0.4346],
          [-0.1358]]]], grad_fn=<TransposeBackward0>)
output:
 tensor([[[ 0.0766,  0.0755, -0.0321],
         [ 0.0311,  0.1048, -0.0368],
         [ 0.0165,  0.1088, -0.0409],
         [-0.0470,  0.0841, -0.0825],
         [-0.1018,  0.0327, -0.1292],
         [-0.1060,  0.0508, -0.1246]]], grad_fn=<ViewBackward0>)
output.shape:
 torch.Size([1, 6, 3])
```

Khác biệt chính là sau khi hoàn tất `context_vec`, chúng ta cần kết hợp kết quả của tất cả head vì có thể thấy shape của `context_vec` là `[1, 6, 3, 1]`. Nó có nhiều hơn một dimension so với cần thiết. Chúng ta muốn nén nó xuống chỉ còn `[1, 6, 3]`. Đó là điều dòng này thực hiện:

```python
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
```

Có một vài lý do cho optional linear projection:

- Nó cho phép chúng ta biến đổi embedding dimension thành output có embedding dimension khác để đưa vào phần sau của LLM nếu cần.

- Nó đóng vai trò một learnable transformation. Nghĩa là chúng ta có thể training projection này để học cách trộn và đặt weight cho thông tin từ mỗi head.

Vậy là phần giải thích cơ bản về cách attention mechanism hoạt động đã kết thúc. Tất nhiên, attention mechanism được sử dụng trong các LLM gần đây còn có nhiều khía cạnh khác, nhưng đây là nền tảng có thể giúp bạn bắt đầu và đi sâu hơn vào việc tìm hiểu tất cả những component này.

Cảm ơn bạn đã đọc và tôi hy vọng hôm nay bạn đã học được điều gì đó!
