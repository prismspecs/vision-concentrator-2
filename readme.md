# Vision Morphing App

A web application that generates a morphing video between multiple visions (text prompts) using **Stable Diffusion XL Turbo (SDXL Turbo)**. The application provides a Flask web interface where users can input their visions, and it generates images by interpolating between those visions and compiles them into a morphing video.

## Features

- **Web Interface**: User-friendly web interface built with Flask.
- **Vision Input**: Accepts multiple visions (text prompts) from the user.
- **Image Generation**: Generates images by interpolating between the visions using SDXL Turbo.
- **Video Creation**: Compiles generated images into a morphing video using MoviePy.
- **Dockerized Deployment**: Runs inside a Docker container with GPU acceleration.

## Requirements

### Hardware

- **NVIDIA GPU**: GPU with CUDA support (ideally with at least 12GB VRAM).

### Software

- **Operating System**: Ubuntu 22.04 or compatible Linux distribution.
- **Docker**: Installed on the host machine.
- **NVIDIA Container Toolkit**: Installed for GPU support in Docker.

### Model Files

- **SDXL Turbo Model File**: `sd_xl_turbo_1.0_fp16.safetensors` placed inside `app/models/`.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vision-morph-app.git
cd vision-morph-app
```

### 2. Place the SDXL Turbo Model File

Ensure that you have the `sd_xl_turbo_1.0_fp16.safetensors` model file. Place it inside the `app/models/` directory:

```
vision-morph-app/
├── Dockerfile
└── app/
    ├── models/
    │   └── sd_xl_turbo_1.0_fp16.safetensors
    └── ... (other files)
```

### 3. Build the Docker Image

```bash
sudo docker build -t vision-morph-app .
```

### 4. Run the Docker Container

```bash
sudo docker run --gpus all -p 5000:5000 -v $(pwd)/app/:/app/ vision-morph-app
```

## Usage

1. **Access the Web Interface**: Open your web browser and navigate to `http://localhost:5000`.
2. **Enter Visions**: Input your visions (text prompts) separated by commas.
3. **Generate Video**: Click **"Generate Video"**.
4. **Processing**: Wait for the application to process your input.
5. **View Result**: The resulting morphing video will be displayed on the results page.

## Project Structure

```
vision-morph-app/
├── Dockerfile
└── app/
    ├── app.py
    ├── generate_images.py
    ├── generate_video.py
    ├── models/
    │   └── sd_xl_turbo_1.0_fp16.safetensors
    ├── templates/
    │   ├── index.html
    │   └── result.html
    └── static/
        └── (generated videos and images)
```

## Dependencies

The application relies on the following Python packages (installed within the Docker container):

- `torch==2.0.1+cu118`
- `diffusers>=0.19.0`
- `transformers`
- `moviepy`
- `flask`
- `tqdm`
- `numpy`
- `accelerate`
- `safetensors`
- `xformers==0.0.20`

## Dockerfile Overview

The `Dockerfile` sets up the environment with all necessary dependencies, including:

- NVIDIA CUDA base image with Ubuntu 22.04 and cuDNN 8.
- Installation of Python 3.10 and pip.
- Installation of PyTorch with CUDA support.
- Installation of required Python packages.
- Copying the application code into the Docker image.
- Exposing port 5000 for the Flask app.
- Command to run the Flask application.

## Notes

- **Model Licensing**: Ensure that you have accepted the license agreement for the SDXL Turbo model and have the rights to use it.
- **Memory Requirements**: The application uses `xformers` for memory-efficient attention. Ensure it is properly installed to reduce memory usage.
- **First Run**: The initial run may take some time as the model is loaded into memory.
- **GPU Usage**: Make sure your NVIDIA drivers and the NVIDIA Container Toolkit are properly installed and configured.

## Troubleshooting

- **ModuleNotFoundError for `xformers`**:
  - Ensure `xformers` is installed and compatible with your environment.
  - Check the installation logs during the Docker build for errors.

- **CUDA Out of Memory Errors**:
  - Reduce `num_inference_steps` or `frames_per_transition` in `generate_images.py`.
  - Close other applications that might be using the GPU.

- **Application Not Accessible**:
  - Verify that the Docker container is running.
  - Ensure ports are correctly mapped (`-p 5000:5000`).

- **File Not Found Errors**:
  - Ensure all files are correctly placed and paths are properly set in your code and Dockerfile.

## Customization

- **Adjust Parameters**:
  - **`frames_per_transition`**: Increase for smoother transitions (will require more processing time).
  - **`num_inference_steps`**: Adjust to balance between image quality and generation speed.
  - **`guidance_scale`**: Modify to control how closely images adhere to the prompts.

- **Modify Web Interface**:
  - Customize the HTML templates in `app/templates/` to change the appearance or add functionality.

## Security Considerations

- **Licensing Compliance**: Ensure compliance with the licenses of all models and libraries used.
- **Public Deployment**:
  - Add authentication and secure connections if deploying the application publicly.
  - Use a production-grade WSGI server instead of the Flask development server.

## License

This project is provided for educational purposes. Please ensure that you comply with all licenses of the models and libraries used.

---

**Disclaimer**: The SDXL Turbo model and associated files are provided by Stability AI and may have specific licensing terms and usage restrictions. Ensure that you have the proper rights and permissions to use the model in your application.