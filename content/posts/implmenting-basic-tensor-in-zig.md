+++
title = 'Creating Tensor in Zig'
date = 2024-07-30T18:40:12+08:00
draft = true
+++

# Implementing basic tensor in Zig

- The first thing is about Zig is that dynamic allocation of memory is not straight forward as you have to be conscious about every single memory that you work with.

- For generic in Zig, you just use a function to return a struct, this way you can dynamically change the rank of the tensor without having to worry about customizing a tensor each time to fit your shape.
