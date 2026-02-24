# Pavement Texture Evaluator

A project to evaluate and analyze pavement textures via the **TextureLab** Streamlit application.

## 🚀 How to share and deploy this app for others to test (Streamlit Community Cloud)

The easiest and completely **free** way to share this application publicly is using [Streamlit Community Cloud](https://share.streamlit.io). It connects directly to your GitHub repository and automatically updates whenever you push new changes.

### Step-by-Step Deployment Guide

1. **Commit and Push to GitHub**
   Ensure all your latest code (including the `TextureLab/requirements.txt` file) is pushed to your GitHub repository.

2. **Sign up for Streamlit Community Cloud**
   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in using your GitHub account.

3. **Deploy the App**
   - Click the **"New app"** button.
   - Authorize Streamlit to access your GitHub repositories if prompted.
   - Fill in the deployment form:
     - **Repository:** Select this repository (e.g., `callai-Sergio/Pavement-texture-descriptor`)
     - **Branch:** `master` (or `main`)
     - **Main file path:** Type `TextureLab/app.py`
   - Click **"Deploy!"**

4. **Wait for Build**
   Streamlit will now provision a server, install the dependencies listed in `TextureLab/requirements.txt`, and launch the app. This usually takes 1-3 minutes. 

5. **Share the Link!**
   Once deployed, you will get a public URL (e.g., `https://pavement-texture.streamlit.app`) that you can send to anyone to test the application directly in their browser without installing Python.

*(Note: The Community Cloud has a 1GB memory limit. For extremely large LAZ files > 200MB, users might experience slow downs, but it works perfectly for standard pavement scans).*
