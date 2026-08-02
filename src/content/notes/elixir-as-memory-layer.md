---
title: Elixir as memory layer
published: 2026-08-02
---

- Injection of context on the go with concurrency, not afraid of breaking output
- Imagine there is a central memory context on the supervisor / main process
- Then, as we work on things, the robot will spawn thread to understand image / audio / blah blah
- Once the process return back with interpretation, they will inject context into the the main process memory
- Since Elixir works on concurrency, we can continuously update this "current" memory context, while also sending them to other threads to be processed for long-term memory for retrieval later on. We don't have to care about how it is done, because it is the BEAM VM.
- Since each talk / chat is a process by itself, we can just talk to agent while it is doing other task, it basically come free out of the box without needing any weird sub-agent orchestration system.
- Then, an agent can go off to answer a question with the current memory in a seperate process, deciding to spawn even more process to call tools as needed to answer.
- It can do that thing where it says that it keeps thinking also until the other process is done.

