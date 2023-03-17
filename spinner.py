import os
import sys
import time
import platform
import threading

globe_frames = [
    "⣼",
    "⣹",
    "⢻",
    "⠿",
    "⡟",
    "⣏",
    "⣧",
    "⣶"
]

def clear_terminal():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def animate_spinning_globe():
    while True:
        for frame in globe_frames:
            clear_terminal()
            print(frame)
            time.sleep(0.085)

def start_animation():
    animation_thread = threading.Thread(target=animate_spinning_globe)
    animation_thread.daemon = True  # Set as a daemon thread to exit when the main program exits
    animation_thread.start()
