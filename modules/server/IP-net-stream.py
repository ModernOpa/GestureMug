import cv2
import socket
import pickle
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999)) 
server_socket.listen(5)
print("[WIN11-VM] Ожидание подключения от WSL2 на порту 9999...")

cap = cv2.VideoCapture(0) # Открывает камеру, которая работает в твоей виртуальной Win11

try:
    while True:
        client_socket, addr = server_socket.accept()
        print(f"[WIN11-VM] WSL2 успешно подключилась: {addr}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            data = pickle.dumps(frame)
            message = struct.pack("Q", len(data)) + data
            try:
                client_socket.sendall(message)
            except (ConnectionResetError, BrokenPipeError):
                break
except KeyboardInterrupt:
    print("Остановка.")
finally:
    cap.release()
    server_socket.close()
