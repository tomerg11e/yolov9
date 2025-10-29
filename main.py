# import torch
# print(torch.cuda.is_available(), torch.cuda.device_count())

def run_converted_inference():
    import detect

    params = {
        # 'source': 'dataset/images/train/frame_000003.jpg',
        'source': 'dataset/images/train/t3_png.rf.3cf6452aa5948055a40ad797fc4c6cb7.jpg',
        'imgsz': (640, 640),
        'device': 0,
        'weights': 'checkpoints\yolov9-c-converted.pt',
        'name': 'yolov9_c_640_converted'
    }
    detect.run(**params)

def run_overfit():
    import train_dual
    params = {
        'workers': 8,
        'device': 0,
        'batch_size': 16,
        'data': 'dataset_overfit_test/data.yaml',
        'imgsz': 640,
        'cfg': 'models/detect/yolov9-c.yaml',
        'weights': '',
        'name': 'overfit_yolov9-c',
        'hyp': 'hyp.scratch-high.yaml',
        'min_items': 1,
        'epochs': 10,
        'close_mosaic': 0,
        # 'augment': False
    }
    train_dual.run(**params)

def run_training():
    import train_dual
    params = {
        'workers': 8,
        'device': 0,
        'batch_size': 16,
        'data': 'dataset/data.yaml',
        'imgsz': 640,
        'cfg': 'models/detect/yolov9-c.yaml',
        'weights': '',
        'name': 'yolov9-c',
        'hyp': 'hyp.scratch-high.yaml',
        'min_items': 1,
        'epochs': 5,
        'close_mosaic': 1
    }
    train_dual.run(**params)

def run_inference():
    import detect_dual

    params = {
        # 'source': 'dataset/images/train/frame_000003.jpg',
        'source': 'dataset/images/train/t3_png.rf.3cf6452aa5948055a40ad797fc4c6cb7.jpg',
        'imgsz': (640, 640),
        'device': 0,
        'weights': 'checkpoints/yolov9-c.pt',
        'name': 'yolov9_c_640_detect'
    }
    detect_dual.run(**params)

def main():
    # run_inference()
    # run_converted_inference()
    # run_training()
    run_overfit()
    pass

if __name__ == "__main__":
    main()