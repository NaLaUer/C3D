import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path

"""
    clip_len 压缩照片帧数
"""
class VideoDataset (Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        #  root_dir = '/home/sky/PycharmProjects/UCF101/ucf'
        #  output_dir = '/home/sky/PycharmProjects/UCF101/ucf_split'
        #  folder = '/home/sky/PycharmProjects/UCF101/ucf_split/train'
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # 图片相关尺寸
        self.resize_height = 128
        self.resize_width  = 171
        self.crop_size = 112

        # 1 判断数据集是否存在
        if not self.check_integrity():
            raise RuntimeError('数据集不存在.')

        # 视频 => 图像  && 训练集测试集切分
        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, only once.'.format(dataset))
            # 创建训练集、测试集、验证集
            self.preprocess()

        self.fnames, labels = [], []
        # folder = '/home/sky/PycharmProjects/UCF101/ucf_split/train'
        # self.fnames = '/home/sky/PycharmProjects/UCF101/ucf_split/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03'
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        # 判断长度
        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        # 字典, 给label打标签
        self.label2index = {label : index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        # 字典
        if dataset == "ucf101":
            if not os.path.exists('ucf_labels.txt'):
                with open ('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')
        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')

    def __len__(self):
        return  len(self.fnames)


    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)
    # file_dir = '/home/sky/PycharmProjects/UCF101/ucf_split/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03'
    # frames = '/home/sky/PycharmProjects/UCF101/ucf_split/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03/00001.jpg'
    """
        合帧 16 3*171*128  =>  16*3*171*128
    """
    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    """
        随机切割操作 
        source : two-stream-action-recognition-master
    """
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    # 判断是否存在原始文件
    def check_integrity (self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    """
        function : 判断数据是否切分完毕
            1. output_dir是否存在
            2. output_dir/train是否存在
            3. 图片大小是否符合
    """
    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                          sorted(
                                              os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[
                                              0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True


    """
        function : 创建文件夹
        input   : video
        output ： pic
    """
    def preprocess (self):
        # 1 判断输出文件夹是否存在
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))
        # 2 切分数据的训练集与测试集
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            # file_path = /home/sky/PycharmProjects/UCF101/ucf/MoppingFloor
            video_files = [name for name in os.listdir(file_path)]
            # video_files = v_MoppingFloor_g02_c05.avi
            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir   = os.path.join(self.output_dir, 'val', file)
            test_dir  = os.path.join(self.output_dir, 'test', file)
            #train_dir =  /home/sky/PycharmProjects/UCF101/ucf_split/train/MoppingFloor

            # 3 创建文件夹，用于存放每个video的图像
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    """
        video = v_MoppingFloor_g03_c02.avi
        action_name  = MoppingFloor
        save_dir = /home/sky/PycharmProjects/UCF101/ucf_split/train/MoppingFloor
        
        source : cv2 example
    """
    def process_video (self, video, action_name, save_dir):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir (os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture (os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=16, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break