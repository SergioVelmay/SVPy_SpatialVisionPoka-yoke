# SVPy | Spatial Vision Poka-yoke <a id="top"></a>

- [DepthAI Setup](#depthai-setup)
    - [Links](#depthai-setup-links)
    - [Windows](#depthai-setup-windows)
    - [Ubuntu](#depthai-setup-ubuntu)
    - [Raspberry](#depthai-setup-raspberry)
- [SVPy App Run](#svpy-app-run)
    - [NNet Pipelines](#svpy-app-run-pipelines)
    - [Application GUI](#svpy-app-run-application)
    - [Spatial Depth](#svpy-app-run-spatial)

## DepthAI Setup <a id="depthai-setup"></a>

[⇧](#top)

### Links <a id="depthai-setup-links"></a>

- [GitHub Repository](https://github.com/luxonis/depthai-python)

- [Documentation](https://docs.luxonis.com/en/latest/)

- [Welcome to Gen2 API](https://docs.luxonis.com/projects/api/en/latest/)

- [Installation](https://docs.luxonis.com/projects/api/en/latest/install/)

- [Supported Platforms](https://docs.luxonis.com/projects/api/en/latest/install/#supported-platforms)

[⇧](#top)

### Windows <a id="depthai-setup-windows"></a>

- Windows 10 Pro

- [Install the necessary dependencies](https://docs.luxonis.com/projects/api/en/latest/install/#windows)

1. Run Windows PowerShell as administrator

    ![Windows PowerShell: Run as Administrator](./Images/windows_powershell.jpg)

2. Install Chocolatey
    ```
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    ```

3. Close the PowerShell and then re-open another PowerShell as administrator.
    ```
    WARNING: It's very likely you will need to close and reopen your shell before you can use choco.

    You may need to shut down and restart powershell and/or consoles first prior to using choco.
    ```

4. Install Python and PyCharm
    ```
    choco install cmake git python pycharm-community -y
    ```

5. Install the DepthAI library
    ```
    python -m pip install depthai
    ```

6. Clone the repository
    ```
    cd C:\...\SVPy\OAK-D

    git clone https://github.com/luxonis/depthai-python.git

    cd depthai-python
    ```

7. Install the requirements
    ```
    cd examples

    python install_requirements.py
    ```

8. Test the installation
    ```
    Connect the OAK-D device to a USB 3 port.

    python 01_rgb_preview.py
    ```

[⇧](#top)

### Ubuntu <a id="depthai-setup-ubuntu"></a>

- Ubuntu 20.04 LTS 64-bit

- [Install the necessary dependencies](https://docs.luxonis.com/projects/api/en/latest/install/#ubuntu)

1. Install Python 3.6
    ```
    sudo apt update

    sudo add-apt-repository ppa:deadsnakes/ppa

    sudo apt-get install python3.6
    ```

2. Clone repository
    ```
    git clone https://github.com/SergioVelmay/SVPy_SpatialVisionPoka-yoke.git
    ```

3. Change folder
    ```
    cd SVPy_SpatialVisionPoka-yoke/
    ```

4. Create virtual environment
    ```
    sudo apt install python3.6-venv

    python3 -m venv venv
    ```

5. Active environment
    ```
    source venv/bin/activate
    ```

6. Install dependences
    ```
    sudo apt install python3-pip

    pip install upgrade pip

    pip install opencv-python

    pip install depthai

    pip install pillow

    sudo apt-get install python3-pil python3-pil.imagetk

    pip install openvino

    pip install openvino-dev
    ```

8. Change folder
    ```
    cd SVPy/
    ```

9. Run the project
    ```
    python3 svpy.py
    ```

[⇧](#top)

### Raspberry <a id="depthai-setup-raspberry"></a>

- Raspberry Pi OS 32-bit

- [Install the necessary dependencies](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os)

1. Install OpenVINO toolkit for Raspbian OS
    - [l_openvino_toolkit_runtime_raspbian_v.tgz](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/)

    - [Install OpenVINO for Raspbian OS package](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html#install-package)

2. Clone repository
    ```
    git clone https://github.com/SergioVelmay/SVPy_SpatialVisionPoka-yoke.git
    ```

3. Change folder
    ```
    cd SVPy_SpatialVisionPoka-yoke/
    ```

4. Create virtual environment
    ```
    sudo apt install python3.6-venv

    python3 -m venv venv
    ```

5. Active environment
    ```
    source venv/bin/activate
    ```

6. Install dependences
    ```
    pip install upgrade pip

    pip install depthai

    sudo apt-get install python3-pil.imagetk

    sudo pip3 install gpiozero
    ```

8. Change folder
    ```
    cd SVPy/
    ```

9. Run the project
    ```
    python3 svpy.py
    ```

[⇧](#top)

## SVPy App Run <a id="svpy-app-run"></a>

```
cd SVPy_SpatialVisionPoka-yoke/

cd SVPy/
```

[⇧](#top)

### NNet Pipelines <a id="svpy-app-run-pipelines"></a>

- Part Count Detection
    ```
    python3 blob_part_count_detection.py
    ```

- Multilabel Classification
    ```
    python3 blob_multilabel_classification.py
    ```

- Objects Detection
    ```
    python3 blob_objects_detection.py
    ```

- Multiclass Classification
    ```
    python3 blob_multiclass_classification.py
    ```

- Hidden part Detection
    ```
    python3 blob_hidden_part_detection.py
    ```

- O-Ring Classification
    ```
    python3 blob_oring_classification.py
    ```

- Gloves Classification
    ```
    python3 blob_gloves_classification.py
    ```

[⇧](#top)

### Application GUI <a id="svpy-app-run-application-spatial"></a>

- Main Application GUI
    ```
    python3 svpy.py
    ```

- Main App GUI without GPIO
    ```
    python3 svpy_no_gpio.py
    ```

[⇧](#top)

### Spatial Depth <a id="svpy-app-run"></a>

- Part Count Levels
    ```
    python3 svpy_depth.py 0
    ```

- Level #0 to Step #1
    ```
    python3 svpy_depth.py 1
    ```

- Level #0 to Step #2
    ```
    python3 svpy_depth.py 2
    ```

- Level #0 to Step #3
    ```
    python3 svpy_depth.py 3
    ```

- Level #3 to Step #4
    ```
    python3 svpy_depth.py 4
    ```

- Level #4 to Step #5
    ```
    python3 svpy_depth.py 5
    ```

- Level #4 to Step #6
    ```
    python3 svpy_depth.py 6
    ```

- Level #5 to Step #7
    ```
    python3 svpy_depth.py 7
    ```

- Level #2 to Step #8
    ```
    python3 svpy_depth.py 8
    ```

[⇧](#top)