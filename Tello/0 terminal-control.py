"""Tello Control Demo
Provides direct control of Tello drone for easy testing.

Full command list at:
- (Tello EDU) https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
- (Tello TT) https://dl.djicdn.com/downloads/RoboMaster+TT/Tello_SDK_3.0_User_Guide_en.pdf

If you are using the Tello TT but are unable to use the EXT commands 
(e.g. `EXT tof?` to get distance sensor reading), send command `sdk?`.
If the response is not 30, you need to update the drone firmware using the Tello app.

The first command you should usually send is "command" to enter SDK mode.
"""

import threading 
import socket

host = ''
port = 9000
locaddr = (host, port) 

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

def recv():
    while True: 
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nThere was an exception. Exit...\n')
            break

print ('\r\n\r\nTello Python3 Demo.\r\n')
print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')
print ('end -- quit demo.\r\n')

#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()

while True: 
    try:
        msg = input("");
        if not msg:
            break  
        if 'end' in msg:
            print ('...')
            sock.close()  
            break
        # Send data
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)

    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()  
        break
