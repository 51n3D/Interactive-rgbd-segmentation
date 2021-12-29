Repo: https://github.com/51n3D/Interactive-rgbd-segmentation

Train usage:
python3 train.py [dataset root dir] [checkpoint files root dir]

GUI usage:
cd gui && python3 app.py app.py [-h] -i IMG [-m MODEL] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -i IMG, --img IMG     image name (without _leftImg8bit.png)
  -m MODEL, --model MODEL
                        stored model parameters
  -d, --debug           debug mode
