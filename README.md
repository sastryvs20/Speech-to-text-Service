# 🗂️ Transcription Service

## 1. 📁 Create & Activate a Python Virtual Environment

### 1-a Create the environment (Python 3.10 recommended)

```bash
python3.10 -m venv transcribe_env

```

### 1-b Activate the environment

**Linux / macOS**

```bash
source transcribe_env/bin/activate

```

**Windows**

```powershell
transcribe_env\Scripts\activate

```

----------

## 2. 📦 Manually Install Core Dependencies

```bash
pip install numpy==1.26.4
pip install typing_extensions==4.14.0
pip install sox==1.5.0
pip install packaging==25.0
pip install wheel==0.45.1
pip install ffmpeg==1.4
pip install "fastapi>=0.111"
pip install "uvicorn[standard]>=0.29"
pip install python-multipart==0.0.20
```

> 💡 **Why manual?**  
> Installing these individually avoids version clashes with `nemo_toolkit` and `vllm`.

----------

## 3. 🔊 Install NeMo Toolkit & vLLM (with legacy resolver)
    
**3a**.  **Install both packages using the legacy resolver:**
    
    
    pip install --use-deprecated=legacy-resolver -r requirements.txt
    
    

## 4. 🚀 Run the Transcription Service

### 4-a Start the API normally
   ```bash
cd Whisper-Service/Whisper-Service
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```
   ### 4-b Run in the background with `nohup`
```bash
cd Whisper-Service/Whisper-service
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8002 &
