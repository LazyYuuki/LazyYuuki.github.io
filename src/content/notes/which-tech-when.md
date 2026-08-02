---
title: Which tech when?
published: 2026-08-02
---

Question to ask when starts a project:
- Does it needs to serve a lot of user or a few single user?
- Does it needs to keep a lot of long-lived connection?
- Does it needs to process a lot of request per second?
- Does it needs high fault-tolerance and scaling?
- Does it optimize for concurrency or for speed?

### In my opinion:

#### Zig, Rust, C, Go 
- Embedded system
- Small and fast single user app
- Reliable process
- A lot of HTTP request per second
- SPEEEDDDD

#### Elixir
- Massive concurrency
- Chat systems
- Real-time related stuff like dashboard, messaging, collaborative app

#### Python
- Just for experimenting and trial
- Should never build real system with it

