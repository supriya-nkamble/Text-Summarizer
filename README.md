# Text-Summarizer

## Workflows

1. update config.yaml file
2. update params.yaml
3. update entity
4. update configuration manager in src/config
5. update the components
6. update the pipeline
7. update the main.py
8. update the app.py

## 🚀 How to Run?

### 🔧 Steps

#### 📥 Clone the Repository

```bash
git clone https://github.com/entbappy/End-to-end-Text-Summarization
cd End-to-end-Text-Summarization
```

#### 🐍 Step 01 - Create a Conda Environment

```bash
conda create -n summary python=3.8 -y
conda activate summary
```

#### 📦 Step 02 - Install the Requirements

```bash
pip install -r requirements.txt
```

#### ▶️ Run the App

```bash
python app.py
```

Then open your **localhost** in the browser to access the application.

---

## 👨‍💻 Author

**Supriya Kamble**  
AI Engineer

---

## ☁️ AWS CICD Deployment with GitHub Actions

### 📝 Prerequisites

1. **Login to AWS Console**
2. **Create an IAM User** for deployment with specific access:
   - **EC2 Access**: Virtual Machine service
   - **ECR Access**: Elastic Container Registry to store Docker images

---

### ⚙️ Deployment Description

1. Build Docker image of the source code
2. Push your Docker image to **ECR**
3. Launch your **EC2 instance**
4. Pull your image from **ECR** in EC2
5. Launch the Docker container in EC2

---

### 🔐 IAM Policies Required

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

---

### 🐳 Create ECR Repository

- Save the URI:

```
566373416292.dkr.ecr.us-east-1.amazonaws.com/text-s
```

---

### 💻 Create EC2 Machine (Ubuntu) and Install Docker

#### Optional

```bash
sudo apt-get update -y
sudo apt-get upgrade
```

#### Required

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 🤖 Configure EC2 as Self-Hosted Runner

1. Go to: GitHub **Repo > Settings > Actions > Runners**
2. Click **New self-hosted runner**
3. Choose OS: **Linux**
