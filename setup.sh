sudo apt update
sudo apt install ffmpeg    # install ffmpeg from the repository

git clone https://github.com/nlohmann/json.git      # Cloning the json repository from github

cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4          # Downloading the videos required for the RI's

cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name person-vehicle-bike-detection-crossroad-0078         # Downloading the person-vehicle-bike-detection model

