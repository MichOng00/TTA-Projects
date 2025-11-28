# does not work

import socket
import sys

def send_command(drone_ip, ssid, password, new_ip):
    # Tello uses UDP port 8889
    tello_address = (drone_ip, 8889)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Step 1: enter command mode
    sock.sendto(b'command', tello_address)

    # Step 2: set station mode
    sock.sendto(f'station:{ssid} {password}'.encode(), tello_address)

    # Step 3: set static IP
    sock.sendto(f'set ipaddr {new_ip}'.encode(), tello_address)

    print("Commands sent. Reboot the drone and reconnect to the new network.")
    sock.close()

# Example usage:
# python set_ip.py 192.168.10.1 MySSID MyPassword 192.168.1.100
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python set_ip.py <tello_ip> <wifi_ssid> <wifi_password> <new_static_ip>")
    else:
        send_command(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
