# Server (server.py)
import socket

def run_server():
    host = ''  # Listen on all available interfaces
    port = 8080  # Choose a port number

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    server_socket.settimeout(5)

    print(f"Server listening on {host}:{port}")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
        except TimeoutError:
            continue
        print(f"Connection from {client_address}")

        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            message = data.decode()
            print(f"{message}")

            response = f"Echo: {message}"
            client_socket.send(response.encode())

        client_socket.close()
        print(f"Connection closed with {client_address}")

if __name__ == "__main__":
    run_server()