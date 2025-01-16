+++
title = 'Build a LLM from scratch: Attention Mechanism'
date = 2024-11-18T20:00:00+08:00
lastmod = 2025-01-11T21:00:00+08:00 
toc.enable = true
tags = ["ai", "engineer"]
series = ["LLM from scratch"]
summary = "In this blog post, I attempt to distill what I have learn about building an attention mechanism for LLM in an easy to digest format. It is heavily inspirted by the content of Sebastian Raschka's Build a Large Language Model (From Scratch) with extra notation to make the content more clear to newcomer."
+++

Although I have been working with LLMs since the GPT-2 days, I actually have never fully built a LLM from scratch myself. A lot of my knowledge stems from my curiosity and so a lot of self-learning was conducted without any proper systematic structure. Even in my university course, the detailed of how to build a full fletch LLM is only glossed over, although I do gain a rather deep understanding of how tokenization works thank to having to build one myself. In this (hopefully) series of blog post, I will try to distill what I have learnt with the hope that it would be useful for someone else in the future to learn from too!

> [!NOTE]
> 
> Disclaimer: This blog post is heavily inspired by the content of Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).
>
> Please give it a read if you really want to go through the process step by step and have an even more indepth look at it.


## 1. Basic input

Before we get to how an attention block works, let first defined our input that we are going to work with.

Taken this following toys embedding as an example:

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

This is a tensor that shows the 3-dimensional word embedding of 6 tokens of the following input:

> Your journey starts with one step

A word embedding is a vector representation of data in a continuous, high-dimensional space that captures semantic relationships.

```python
[0.43, 0.15, 0.89] # this is the embedding of the token "Your" 
                   # with an embedding dimension of 3
```

If you perform a dot product between two embedding, you will get a scalar value that will represent how similar they are to each other. This is an important concept that explains why the attention mechanism is constructed the way it is.

Usually, an embedding dimension will be much higher than this (e.g. GPT-3.5 with 12,288 dimensions). However, for demonstration purpose, we shrink the embedding to only 3-dimmension.

In this articles, we will refer to the whole tensor as the input sequence, and the token that we are currently working on as the query token. This is because LLM don't ingest the whole sequence as once but rather sequentially ingesting each token in the sequence so that it can generate a sequentially coherent output sequence.

## 2. Overview of the Attention Mechanism

The objective of the attention mechanism is to calculate the **context vector** of a query token.

A **context vector** purpose is to create enriched representations of each token in an input sequence by incorporating information from all other token in the sequence. The LLM will use trainable weight matrixes to learn to construct these context vectors so that they are relevant for the LLM to generate the next token.

To get the context vector, we need to first calculate the **attention score**. It shows how relevant each of the tokens in an input sequence is relative to the query token. The higher the score, the more relevant, and the more the query token should pay "attention" to them. 

The attention score is calculated by performing a dot product between the query token embeddings and all the other tokens in the sequence (including itself).

```python
query = inputs[1]  # "journey" as the query token

attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)

print(attn_scores)
```

```output
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

We then normalise this attention score through a softmax function to turn this into an **attention weight**. This will make sure that all the elements in the tensor sum up to 1. 

```python
attn_weights = torch.softmax(attn_scores, dim=0)

print("Attention weights:", attn_weights)
print("Sum:", attn_weights.sum())
```

```output
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
```

The reason we use softmax is to better manage extreme values and offers more favorable gradient properties during training.

After we get the attention weight, we can multiply this with the embedding of all the tokens in the sequence and sum them up to get the context vector of the query token.

```python
context_vec = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec += attn_weights[i]*x_i

print(context_vec) 
```

```output
tensor([0.4419, 0.6515, 0.5683])
```

As we can see, the context vector has exactly the same dimension as the token embedding. It is just an enriched version of the original embedding essentially.

## 3. Trainable Attention Block

### 3.1. What is new?

The example above is a simplified calculation to illustrate the general concept of attention mechanism. As you can clearly see, it does not have any trainable weight, which is what we are going to add in this section.

To do the same thing as we did above, we needs 3 new trainable weight matrixes name:

- query
- key
- value

To mirror what we have described above. With each of the input tokens' embedding:

- multiply with the query weight matrix, we get the query vector. The query vector replaces the query embedding in the simplified example. 

- multiply with the key weight matrix, we get the key vector. In the simplified example, the query and key vector value for the query token is exactly the same value, but here they are not necessarily the same. The key value is particularly use to calculate the attention score and attention weight only. Analogy wise, you could think of it as a vector used solely to measure how relevant this token is to the query token.

- multiply with the value matrix, we get the value vector. In the simplified example, the token embeddings is the value vector. In this version, the value vector is what you used to multiply with the attention weight to get the context vector. Analogy wise, think of the value vector as the actual content representation of the token that would be used to calculate the context vector.

The idea is that these trainable matrixes will change as the LLM learn. Once the model determines which **keys** (part of the sequence) are most relevant to the **query** (the focus item), it will retrieve the corresponding **values** (the numerical values of the token).

Comparatively to a database system, we can think of the:

- query as the search input (e.g: yellow)
- key as the indexing of the item (e.g: yellow, gold, ... )
- value as the actual value of the item (e.g: #FFFF00, #FFD700)

### 3.2. The basic idea 

To be able to get these query, key, and value vector (hence forth refer to as qkv), we train the qkv weight matrixes respectively. So it will look something like this:

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

The output is as follow:

```output
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

In practices, we want to receive the same qkv vector dimension as the input embedding dimension, so in this instance we are expecting a 3 x 3 weight matrixes for qkv respectively. 

Let go through the process to calculate the required qkv vector:

```python
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("Query vector\n", query_2)
print("Key vector\n", key_2)
print("Value vector\n", value_2)
```

```output
Query vector
 tensor([0.8520, 0.4161, 1.0138], grad_fn=<SqueezeBackward4>)
Key vector
 tensor([0.7305, 0.4227, 1.1993], grad_fn=<SqueezeBackward4>)
Value vector
 tensor([0.9074, 1.3518, 1.5075], grad_fn=<SqueezeBackward4>)
```

With these vectors, we just repeated the step in section 2 to retrieve all the context vectors with respect to the query "journey". 

With that understanding in mind, we realise that actually we can just replace the `nn.Parameter` module with `nn.Linear` because a linear operation with no bias is essentially equal to a matrix multiplication. That being said, `nn.Linear` is also much better at initializing a more stable matrix for computation purpose. Hence, we have the following Single Self-Attention Mechanism.

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

In the forward method, we generates all the qkv vectors for each tokens, and then use them to efficiently calculate all the context vectors for each tokens at once.

### 3.3. Let focus on the present

Right now, the problem is that the query token is paying attention to **ALL** the tokens in an input sequence. This is not ideal, because we don't want the model to pay attention to token that has not appeared yet because intuitively it makes sense NOT to pay attention to things that happened in the future. 

We introduce causal / masked attention, where we masked all the future value with 0 value, so that it is not part of the context vector calculation anymore. 

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

If we run the CausalAttention above:

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

We will get:

``` output
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

Let's break down the how the CausalAttention class works. 

First in the initialisation section, we add 2 new parts:
- `Dropout`
- `register_buffer`

Strictly speaking, you don't need `register_buffer` for the class to work but it does help simplify the process of assigning computation to the proper device. `Dropout` is module that allow you to randomly zeroes (aka drop) some of the elements of the input tensor with a certain probabilities. So if the dropout rate is 0.2. That would means for a tensor with 10 elements, 2 elements will be randomly drop.

Then, we look at the forward method. The input x is actually in the form of a batch data where: 

> x.shape = [num_batch,num_tokens, d_in]

The input x is then put through the `Linear` module to get their corresponding qkv weight matrixes. It is important to note that in Pytorch, a `nn.Module` has the following characteristic after initialisation:

```python
module = nn.Module(foo,bar,baz)

module(x)
# is the same as
module.forward(x)
```

In another words

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

This is a very common pattern in Pytorch but often glossed over by more experienced people when trying to explain common practice of Pytorch. So to avoid confusion, just understand that if a `nn.Module` is initialised, you can call the class like a function, and understand implicitly that it is calling the `forward` method.

The shape that we would get from `keys`, `queries`, `values` is equal to  `[num_batch,num_tokens, d_in]`. We can see this from the output:

```output
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

The reason why both the batch has the exact same value is because we provide as an input: 2 batch with the exact same input. But as can clearly be shown, there are 2 of 6 x 3 matrixes, each belong to a batch respectively.

We then calculate the attention score. We can see that the keys is transposed before performing matrix multiplication with queries. `keys.transpose(1,2)` will look something like this:

```output
tensor([[[ 0.2727,  0.1008,  0.1060,  0.0051,  0.1696, -0.0388],
         [-0.4519, -0.7142, -0.7127, -0.3809, -0.4861, -0.4213],
         [ 0.2216, -0.1961, -0.1971, -0.1557, -0.1597, -0.1501]],
        [[ 0.2727,  0.1008,  0.1060,  0.0051,  0.1696, -0.0388],
         [-0.4519, -0.7142, -0.7127, -0.3809, -0.4861, -0.4213],
         [ 0.2216, -0.1961, -0.1971, -0.1557, -0.1597, -0.1501]]])
```

So effectively, we only transpose the keys vector along its 2nd and 3rd dimension (not 1st and 2nd because Python index start from 0). So if the shape of `keys` is `[2, 6, 3]` then `keys.transpose(1,2)` is `[2, 3, 6]`. This is the same as what we have shown earlier when you do the multiplation between query and key in `SelfAttention`, just repeat it with one more batch of data.

Next, we masked the attention score. As you can see, anything above the diagonal line is `-inf` or negative infinity. This is because we want to normalise the attention score to get the attention weight, but if we just put 0, then the elements above the diagonal line would still have a weightage value due to how `softmax` work. We don't want that. In order to truly zero out the weight, we need to use `-inf`. 

```output
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

Which result in the following attention weights:

```output
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

Next we apply the dropout rate of 0.2 (or 20%) on the attention weights to get the following:

```output
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

You can see that when the dropout is applied. It will zero out some value, but it also help re-calculate all the attention weight at the same time to make sure that the remaining weight is "heavier" to account for the zeroed out weight.

And finally, we calculate the context vector.

### 3.4. Why stop at one head?

In practice, we use Multi-Head Attention instead of just a single Causal Attention. Multi-Head attention can be analogously describe as a bunch of Single Causal Attention compute in parallele and combined at the end.

Let try to modified the above Causal Attention to arrive at a Multi-Head attention architecture.

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

For the sake of consistency, we will be using the following configuration as input:

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

We have 0 drop out rate and 3 attention heads. The d_out must be divisible by the number of heads, because if not we won't be able to divide the d_out symmetrically to each of the heads. It will be more apparent when we look into the code in a bit. We have also reduced the batch to just 1 input so it is easier to show the results and do demonstration calculation. 

We have also print out the keys' shape because it is stand to follow that queries and values has the exact shame shape, since they have the same transpose. Now let takes a look until the `attn_scores` calculation portion of the `forward` method.

```output
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

As you can very quickly see, we still have a 6x6 attn_score matrixes for each of the 3 attention heads. The difference here is that each of the head captures 1/3 of the information comparatively to the Causal Attention architecture. Since each of the head captures only a portion of the input sequence, they only attend to different subset of the sequence, but collectively they caputre a much more diverse pattern compare to a Single / Causal Attention design.

The next portion follows the same calculation we have shown earlier. They just need to be transpose so that the data is organised in the correct dimension for matrix multiplication.

```output
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

The main difference is that, after we are done with the `context_vec`, we need to combine all the heads' results together because we can see the `context_vec` shape is `[1, 6, 3, 1]`. That is one more dimension than needed. We want it to be compressed to just `[1, 6, 3]`. That is what this line does:

```python
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
```

There are a few reasons for the optional linear projection:

- It is so that we can morph the dimension embedding into an output of different embedding dimension to the later part of the LLM if need to.

- It is acts as a learnable transformation. Meaning we can train this projection to learn how to mix and wieght the information from each head.

And that is the wrap up for the basic explanation of how the attention mechanism works. Of course, there is many more other aspect to the attention mechanism that is used in more recent LLMs, but this is the fundamental that could help you get start and dive deeper into understanding all of these components. 

Thank you for reading and I hope you have learned something today!
