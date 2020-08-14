import torch
import numpy as np
from network import C3D_model
import cv2

torch.backends.cudnn.benchmark = True


"""
    source ： https://github.com/qiaoguan/Fall-detection
"""
def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    print ("device is available : ", torch.cuda.is_available())

    with open("./dataloaders/ucf_labels.txt", 'r') as f:
        class_names = f.readlines()
        f.close()

    # input model
    model = C3D_model.C3D(num_classes = 101)
    # 加载模型
    checkpoint = torch.load(
        './run/run_0/models/C3D-ucf101_epoch-99.pth.tar',
        map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()

    video = './CliffDiving.mp4'
    cap = cv2.VideoCapture(video)
    retaining = True
    # 该参数是MPEG-4编码类型，后缀名为avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output_CliffDiving.avi', fourcc, 20.0, (320, 240))

    clip = []

    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda()

            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (138, 43, 226), 2)
            cv2.putText(frame, "Prob: %.4f" % probs[0][label], (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (138, 43, 226), 2)
            clip.pop(0)

        if retaining == True:

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
