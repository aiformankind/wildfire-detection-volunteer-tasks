import os
import datetime
import re
import shutil

pathStr = r"C:\Users\sidta\Desktop\AI For Mankind\alertwildfire"
path = os.walk(pathStr)

detectedStr = r"C:\Users\sidta\Desktop\AI For Mankind\alertwildfire\detected"
detected = os.walk(detectedStr)

inputStr = r"C:\Users\sidta\Desktop\AI For Mankind\alertwildfire\input"
input = os.walk(inputStr)

outputStr = r"C:\Users\sidta\Desktop\AI For Mankind\alertwildfire\output"
output = os.walk(outputStr)


currFile = ""
numFilesCopied = 0

def getTimeStamp(filename):
    temp = filename.split("_")
    temp1 = temp[len(temp)-1].split(".")
    return temp1[0]

def getDate(timestamp):
    temp = datetime.datetime.fromtimestamp(timestamp)
    temp1 = str(temp).split()[0]
    temp1 = re.sub('[-]', '', temp1)
    return temp1


for root, directories, files in os.walk(detectedStr):
    for file in files:
        currFile = file

        timestamp = int(getTimeStamp(currFile))
        date = getDate(int(timestamp))


        for root, directories, files in os.walk(inputStr):
            for directory in directories:
                #  print(directory + " vs " + date)
                 if directory == date:
                    newDirStr = os.path.join(inputStr, directory)
                    
                    for r, d, f in os.walk(newDirStr):
                        for file in f:
                            if file == currFile:
                                fileLoc = os.path.join(newDirStr, file)
                                fileDest = os.path.join(outputStr, file)
                                shutil.copyfile(fileLoc, fileDest)
                                numFilesCopied = numFilesCopied + 1

print(str(numFilesCopied) + " file(s) copied to output folder")
