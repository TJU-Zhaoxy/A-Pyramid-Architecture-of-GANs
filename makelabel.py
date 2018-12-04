import os

origin_dir = "D:/GAN/CACD2000/"
des_dir = "D:/GAN/data/"
imgFiles = [file for file in os.listdir(origin_dir)]

def encodeAge(n):
    if n<=30:
        return 0
    elif n<=40:
        return 1
    elif n<=50:
        return 2
    else:
        return 3


def makeDir():
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)

    for i in range(4):
        new_folder = os.path.join(des_dir, format(i, "<01"))
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

def moveFiles():
    for file in imgFiles:
        lst = file.split("_")

        age = int(lst[0])

        folder = format(encodeAge(age), "<01")
        origin_file = os.path.join(origin_dir, file)
        des_file = os.path.join(des_dir, folder, file)
        os.rename(origin_file, des_file)
