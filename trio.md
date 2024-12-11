# Trio: Simplifying Python Coroutines with Go's Goroutine Style

Python has supported coroutines since versions 3.5 and 3.6.  However, its implementation has been criticized for being unnecessarily complicated.

In contrast, Scheme supports coroutines by providing the `call/cc` function, which captures the current stack and program counter (PC) into a continuation.  Users can leverage `define-macro` to build custom coroutine syntax on top of continuations.

Go, on the other hand, abstracts away low-level concepts like continuations, offering simple and powerful tools: channels and goroutines.

Python introduces new keywords like `async`, which alter the behavior of existing keywords such as `def`, `with`, and `for`.  Additionally, many coroutine-related features are relegated to the `asyncio` package.  While high-level libraries like `fastapi` simplify coroutine usage for work such as HTTP server development, using Python's coroutine primitives directly for complex applications -- such as large language model (LLM) servers with continuous batching -- can be cumbersome.

For more sophisticated tasks, a library that encapsulates the complexity of coroutines into abstractions similar to Go's channels and goroutines is preferable.  Enter `trio`.

As a Go programmer, I found it straightforward to understand Trio by drawing parallels to Go.  Below is a simple producer-consumer example in Go:

```go
package main

func producer() chan int {
    ch := make(chan int)
    go func() {
        for i := 0; i < 5; i++ {
            ch <- i
        }
        close(ch)
    }()
    return ch
}

func main() {
    ch := producer()
    for x := range ch {
        println(x)
    }
}
```

Here's the equivalent implementation in Python using Trio:

```python
import trio


async def producer(nursery):
    s, r = trio.open_memory_channel(0)

    async def _():
        async with s:
            for i in range(5):
                await s.send(i)

    nursery.start_soon(_)
    return r


async def main():
    async with trio.open_nursery() as nursery:
        r = await producer(nursery)
        async with r:
            async for x in r:
                print(x)


trio.run(main)
```

In Go, `chan` is a first-class type representing a blocking queue with a configurable buffer size. Trio offers an analogous abstraction using `MemorySendChannel` (for writing) and `MemoryReceiveChannel` (for reading).

In Go, channels are created using the built-in `make` function.  In Trio, you call `trio.open_memory_channel`.

In Go, functions execute in the main goroutine unless prefixed with `go`, which launches a new goroutine managed by Go's runtime, potentially using multiple OS threads.  Python lacks a built-in coroutine executor, but Trio provides "nursery".  You can launch a coroutine using `nursery.start_soon`.

The Global Interpreter Lock (GIL) in Python prevents true multi-core CPU parallelism for coroutines, unlike Go, which can utilize multiple CPU cores to execute goroutines in parallel.

In Go, closing the channel indicates no more writings will happen.  In Trio, you use `async with`  to manage channel's lifecycle.

Despite Python's verbose coroutine implementation, Trio provides an elegant solution that mimics Go's simplicity and power.  For developers transitioning from Go or working on complex asynchronous tasks, Trio is a robust and intuitive library.

Like Go supports channel of channels, Trio allows us to put channels into channels.  Here is an example Go program:

```go
package main

func number_producer() chan int {
    ch := make(chan int)
    go func() {
        for i := range 5 {
            ch <- i
        }
        close(ch)
    }()
    return ch
}

func channel_producer() chan chan int {
    ch := make(chan chan int)
    go func() {
        for range 3 {
            ch <- number_producer()
        }
        close(ch)
    }()
    return ch
}

func main() {
    ch_of_ch := channel_producer()
    for ch := range ch_of_ch {
        for num := range ch {
            println(num)
        }
        println("---")
    }
}
```

Here is the corresponding Python program that calls Trio:

```python
import trio


async def number_producer(nursery):
    s, r = trio.open_memory_channel(0)

    async def _():
        async with s:
            for i in range(5):
                await s.send(i)

    nursery.start_soon(_)
    return r


async def channel_producer(nursery):
    s, r = trio.open_memory_channel(0)

    async def _():
        async with s:
            for _ in range(3):
                ch = await number_producer(nursery)
                await s.send(ch)

    nursery.start_soon(_)
    return r


async def main():
    async with trio.open_nursery() as nursery:
        ch_of_ch = await channel_producer(nursery)
        async with ch_of_ch:
            async for ch in ch_of_ch:
                async with ch:
                    async for num in ch:
                        print(num)
                print("---")


trio.run(main)
```
