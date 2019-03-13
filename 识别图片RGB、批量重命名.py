from PIL import Image
import os


def RGB_detection_rename(path):
    rename_list = []
    remove_list = []
    files_list = os.listdir(path)
    # print(files_list)
    for file in files_list:
        if file.endswith('.jpg'):
            try:
                with Image.open(file) as img:
                    if img.mode != 'RGB':
                        print(file+', '+img.mode)
                        remove_list.append(file)
                    else:
                        rename_list.append(file)
            except OSError:
                print(file)
                remove_list.append(file)
        else:
            continue

    if remove_list:
        print('存在无法识别的图片')
        for i, filename in enumerate(remove_list):
            remove_name = 'remove_image' + str(i+1) + '.jpg'
            # print(i, filename)
            os.rename(filename, remove_name)
            print('rename_image:{}--->>{}'.format(filename, remove_name))
        print('请手动删除无法识别的图片')
    else:
        print('不存在无法识别的图片文件')

    print('------next------------')

    name = input('重命名图片为:')
    for i, filename in enumerate(rename_list):
        # print(i, filename)
        new_name = name + str(i+1) + '.jpg'
        os.rename(filename, new_name)
        # print('{}--->>{}'.format(filename, new_name))

    return None


if __name__ == "__main__":
    # 获取当前工作路径
    path = os.getcwd()
    # path = 'C:\\Users\\Zilch\\Desktop\\test\\test_image'
    print(path)
    print('----------------------')
    RGB_detection_rename(path)
