import argparse
import os
import cv2
import glob
import numpy as np
import torch

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile, args):
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(cv2.resize(img, (0,0), fx=args.factor, fy=args.factor))
    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_dir, index):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    img_flo = cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, 'flow{}.png'.format(index)), img_flo)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    
    model.to(DEVICE)
    model.eval()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('dir: {}'.format(os.path.join(args.path)))
        
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg')) + \
                    glob.glob(os.path.join(args.path, '*.jpeg')) + \
                    glob.glob(os.path.join(args.path, '*.JPEG'))

        images = sorted(images)
        print(images)
        
        image2 = load_image(images[0], args)
        image4 = load_image(images[1], args)

        img2bgr = cv2.cvtColor(image2[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)
        img4bgr = cv2.cvtColor(image4[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output_path, 'im2.png'), img2bgr)
        cv2.imwrite(os.path.join(args.output_path, 'im4.png'), img4bgr)

        padder = InputPadder(image2.shape)
        image2, image4 = padder.pad(image2, image4)

        _, flow_up = model(image2, image4, iters=32, test_mode=True)
        flow_up = padder.unpad(flow_up)
        viz(padder.unpad(image2), flow_up, args.output_path, 2)
        flow_up = flow_up.detach().cpu()
        torch.save(flow_up, os.path.join(args.output_path, 'f24.pt'))

        _, flow_up = model(image4, image2, iters=32, test_mode=True)
        flow_up = padder.unpad(flow_up)
        viz(padder.unpad(image4), flow_up, args.output_path, 4)
        flow_up = flow_up.detach().cpu()
        torch.save(flow_up, os.path.join(args.output_path, 'f42.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="destination save path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--factor', type=float, default=0.25, help="factor to down-sample/up-sample input images")
    args = parser.parse_args()

    demo(args)