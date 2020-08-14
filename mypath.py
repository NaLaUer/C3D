import os

class Path(object):
    #@staticmethod  该方法不强制要求传递参数
    @staticmethod
    def db_dir (database):
        if database == 'ucf101':
            root_dir = '/home/sky/PycharmProjects/UCF101/ucf'
            output_dir = '/home/sky/PycharmProjects/UCF101/ucf_split'
            return root_dir, output_dir
        else:
            print ("Database {} not available.".format(database))

    def model_dir ():
        return './model/c3dpretrained.pth'


if __name__ == "__main__":
    path = Path()
    root_dir, out_dir = path.db_dir('ucf101')
    print (os.listdir(root_dir))

