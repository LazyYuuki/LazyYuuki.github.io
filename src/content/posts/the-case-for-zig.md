---
title: The case for Zig
description: Early notes on Zig's explicit memory management, safety checks, comptime generics, and custom types.
published: 2025-02-02T17:00:00+08:00
updated: 2025-02-02T17:00:00+08:00
category: Engineering
tags:
  - zig
draft: true
lang: en
---

- Zig don't have any runtime, control all memory management according to user request.
- Zig avoid the problem of undefined behaviour thanks to ability to bound check at both runtime and comptime
- Zig allow for generic through comptime
- Zig allow for method in struct and defining custom type
