def read_last_line(file_path):
    with open(file_path, 'rb') as f:
        f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
        while f.read(1) != b'\n':  # Until EOL is found...
            f.seek(-2, os.SEEK_CUR)  # ...jump back the read byte plus one more.
        return f.readline().decode()  # Read the next line, which is the last line

import os
import time

file_path = 'C:\\Program Files\\Epic Games\\GTAV\\scripts\\VehicleSpeedLog.txt'
while 1:
	os.system("cls")
	time.sleep(0.05)
	last_line = read_last_line(file_path)
	print("KM/h : "+last_line)