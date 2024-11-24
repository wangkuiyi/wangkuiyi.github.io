# A Minimalist HTTP Server Using asyncio

A common misconception about Python is that it’s a low-performance language and unsuitable for server development. However, this notion is contradicted by the fact that renowned large language model serving systems are predominantly written in Python, including [vLLM](https://github.com/vllm-project/vllm/blob/1c445dca51a877ac6a5b7e03ecdb73e0e34d139e/vllm/entrypoints/api_server.py#L14), [SGLang](https://github.com/sgl-project/sglang/blob/84a1698d67d63911e8d1f55c979b00d65d84dc37/python/sglang/srt/server.py#L39), and [JetStream](https://github.com/AI-Hypercomputer/JetStream/blob/d462ca9bbc55531bbe785203cb076e7797250f2a/jetstream/entrypoints/http/api_server.py#L24), all of which utilize a high-level server framework called `fastapi`.

`fastapi` enables programmers to define handler functions that process user requests to different server paths. `uvicorn`, a server library, asynchronously calls these handlers. `uvicorn`’s performance relies on [`asyncio`](https://docs.python.org/3/library/asyncio.html), a feature introduced in Python 3.4 and 3.5. It exposes the Linux system call `epoll` to Python programmers, which is the underlying mechanism behind Nginx’s high-performance asynchronous serving using a single OS thread.

The way asyncio exposes `epoll` is not straightforward. The `epoll` API is quite simple. We associate one or more sockets with an `epoll` file descriptor by calling `epoll_ctl`. Subsequently, each call to `epoll_wait` returns a list of sockets that are ready to be read from or written to. If you’re curious about how a single-threaded C program can use `epoll` to handle concurrent user requests, please refer to [this example](https://github.com/Menghongli/C-Web-Server/blob/master/epoll-server.c).

The package `asyncio` abstracts the `epoll` concepts into *event loop* and *coroutine*. My basic understanding is that an event loop is linked to an `epoll` file descriptor. Typically, we use `asyncio.run` to create the `epoll` file descriptor and then use it to execute a coroutine. Since the given coroutine may spawn other coroutines, and all of them share the same `epoll` file descriptor, it makes sense for the event loop to repeatedly check for ready-to-run coroutines and execute them until they complete or encounter an `await` statement, which put them back to the non-ready mode.

With this likely over-simplified mental model of `asyncio`, I was able to write a single-threaded Python server that can handle client connections concurrently. The program structure closely resembles the C version. Moreover, the calls to `loop.sock_accept`, `loop.sock_recv`, and `loop.sock_sendall` demonstrate that the event loop is linked to sockets, similar to how the C API of `epoll` associates `epoll` file descriptors with sockets.

```python
import asyncio
import socket


async def start_server(client_handler, host="127.0.0.1", port=8888):
    # Step 1: create a socket and bind it to the given port.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(100)  # Set the backlog to 100
    server_socket.setblocking(False)  # Non-blocking mode for asyncio
    print(f"Server listening on {host}:{port}")
    # Step 2: run a loop to accept client connections.
    await accept_clients(server_socket, client_handler)


async def accept_clients(server_socket, client_handler):
    loop = asyncio.get_running_loop()
    while True:
        # Step 1: accept the client connection:
        client_socket, client_address = await loop.sock_accept(server_socket)
        print(f"Connection from {client_address}")
        # Step 2: process the connection asynchronously, so the loop continues without waiting.
        client_socket.setblocking(False)
        asyncio.create_task(client_handler(client_socket, client_address))


async def handle_client(client_socket, client_address):
    try:
        loop = asyncio.get_running_loop()  # Get the event loop created by asyncio.run
        while True:
            # Step 1: read data from the client
            data = await loop.sock_recv(client_socket, 1024)
            if not data:
                break  # Client disconnected
            print(f"Received from {client_address}: {data.decode()}")
            # Step 2: send a response back to the client
            http_response = (
                "HTTP/1.0 200 OK\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                "Content-Length: 13\r\n"  # Length of "Hello, Client!"
                "\r\n"
                "Hello, Client!"
            )
            await loop.sock_sendall(client_socket, http_response.encode())
    except Exception as e:
        print(f"Error with client {client_address}: {e}")
    finally:
        print(f"Closing connection to {client_address}")
        client_socket.close()


if __name__ == "__main__":
    try:  # Run the server
        asyncio.run(start_server(handle_client, "127.0.0.1", 8888))
    except KeyboardInterrupt:
        print("Server shut down")
```

Runing the above program in a terminal session brings up a server listening the local port 8888.  In another terminal session, we could run curl to access the server.  The server would then print the HTTP request sent by curl and responsds with the string `Hello World!`.

The following curl command sends an HTTP GET request:

```shell
curl http://127.0.0.1:8888
```

The server prints the HTTP request:

```plaintext
Received from ('127.0.0.1', 55980): GET / HTTP/1.1
Host: 127.0.0.1:8888
User-Agent: curl/8.7.1
Accept: */*

```

The following command sends an HTTP POST request:

```shell
curl -H 'Content-Type: application/json' \
     -d '{ "title":"foo","body":"bar", "id": 1}' \
	 -X POST http://127.0.0.1:8888
```

The server prints the HTTP request and the posted data:

```plaintext
Received from ('127.0.0.1', 55977): POST / HTTP/1.1
Host: 127.0.0.1:8888
User-Agent: curl/8.7.1
Accept: */*
Content-Type: application/json
Content-Length: 38

{ "title":"foo","body":"bar", "id": 1}
```
