---
title: Thought on Cline [Outdated]
description: Just a few thought on using Cline 
published: 2025-02-06T09:00:00+08:00
draft: false
lang: en
---

So I was curious about how Cline works and whether technology has changed significantly in the last few years or so regarding long context task performing LLM.

And I have come to the simple conclusion that nothing much has changed!

The way cline works is still the exact same way that we allow LLM to have interactivity a from the start of this all which is to stuff everything inside the context, and that is including big big chunk of history context data. 

The main different that has changed is that model now has much bigger context window, much cheaper tokens cost per million and much faster inference speed. But fundamentally, the concept of context stuffing has not changed.

This means that as you work with bigger and bigger codebase and project, there will come a point that one single query is so ludicrously expensive, that it doesn't make sense to get a LLM to do the task for us anymore. Or just simply by the nature of limited context windows, it wouldn't be able to digest the content any further.

That being said, there is quite a bit of improvement in trying to lower the cost of using LLM. One main example of this attempt is in the invention of prompt caching. By the nature of working with LLM in a codebase, when you try to complete 1 task, you would often talk to the LLM model continuously until that tasks is done. Well, then instead of always sending a new prompt with all the old prompt all the time as a new request, why don't we just cache the history in the LLM and keep it warm for the duration that we still uses it for?

Prompt caching works by keeping the cache of all the previous message, and if it has processed a request before, it will keep the portion of the prompt that is exactly the same and use the previously computed portion that matches our new prompt to feed into the inference, instead of having to do computation on the same text prompt again.

The only down size is that prompt cache only stay warm for 5 to 10 minutes, so if you leave a tasks session open for too long on Cline, you would be sending the full context length again to the LLM to be processed, and that is not going to be cheap.

Other than that, the technology is still the same and limited by the context lenght of the model you are using. 

Here is some example data that I have to show how it works:

Every request always "sent" (first time from our prompt, second time from the cache) the full system prompt over, that is why in 2 request, the number of tokens sent is 28.7k but the context window is only 14k ish.

Secondly, you can quickly see that despite sending the exact same system prompt. The first request cost $0.0570, and the second request cost $0.0101. This is thanks to prompt caching on Anthropic service which only charge 10% of the price for reading cache prompt.

Sometimes, when you use Cline, you will also notice that the first request of a new task is really cheap:

That is because the cache is still warm from your previous tasks, so the portion of the system prompt is carried over and save you from having to process the whole prompt again.

It is kinda interesting how I cannot search for any information regarding how this work internally at all.

One other thing to note is that, this would means the more MCP you add to your cline, the more you will be paying per request, even tho you are not using that tools capability. This is something that I think people seems to gloss over. It is not the same as downloading a plugin on extension to your IDE or your app. Once and done.

No. It is more like a subscription. That you will keep continue to pay for every time you send a prompt and for as long as you have it in your list of MCP servers. So be picky with what you want to add to your MCP servers list, and don't be greedy.

