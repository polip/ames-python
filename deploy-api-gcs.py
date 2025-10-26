import subprocess
import os
import sys


PROJECT_ID = 'ames-housing-472418'
REGION = 'europe-west12'
REPO_NAME = "my-python-repo"
SERVICE_NAME = 'ames-fastapi-service'
DOCKERFILE_PATH = 'docker-file-api'

def deploy_with_gcloud():
    """Deploy using gcloud CLI without Python SDK dependencies"""
       
    try:
        print("🚀 Deploying with gcloud CLI...")
        
        # Step 1: Create Artifact Registry repository
        print("📦 Creating Artifact Registry repository...")
        subprocess.run([
            'gcloud', 'artifacts', 'repositories', 'create', REPO_NAME,
            '--repository-format=docker',
            '--location', REGION,
            '--project', PROJECT_ID
        ], check=False)  # Don't fail if already exists
        
        # Step 2: Configure Docker auth
        print("🔐 Configuring Docker authentication...")
        subprocess.run([
            'gcloud', 'auth', 'configure-docker', 
            f'{REGION}-docker.pkg.dev'
        ], check=True)
        
        # Step 3: Build and push image
        image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{SERVICE_NAME}:latest"
        
        print("🔨 Building Docker image...")
        subprocess.run([
            'docker', 'build',
            '-f', DOCKERFILE_PATH,
            '-t', image_uri,
            '.'
        ], check=True)
        
        print("📤 Pushing Docker image...")
        subprocess.run([
            'docker', 'push', image_uri
        ], check=True)
        
        # Step 4: Deploy to Cloud Run
        print("🚀 Deploying to Cloud Run...")
     
        
        subprocess.run([
            'gcloud', 'run', 'deploy', SERVICE_NAME,
            '--image', image_uri,
            '--region', REGION,
            '--platform', 'managed',
            '--allow-unauthenticated',
            '--memory', '4Gi',
            '--timeout','300',
            '--cpu', '2',
            '--project', PROJECT_ID
        ], check=True)

        subprocess.run(['gcloud logs tail',
                        '--follow',
                        '--resource-type=cloud_run_revision',
                        '--resource-labels=service_name=ames-fastapi-service'])
        
        service_url = get_service_url()
        if service_url:
            print(f"✅ Deployment successful!")
            print(f"🌐 Service URL: {service_url}")
            print(f"📖 API Docs: {service_url}/docs")
            print(f"🔍 Health Check: {service_url}/health")
            print(f"🗄️  Model Bucket: gs://scikit-models")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def upload_models_to_storage():
    """Upload model files to Cloud Storage"""
    
    if not os.path.exists('model'):
        print("⚠️  No 'models' directory found. Skipping model upload.")
        return
    
    try:
        print("📤 Uploading models to Cloud Storage...")
        
        # Upload all files in models directory
        subprocess.run([
            'gcloud', 'storage', 'cp', '-r', 'models/*', f'gs://scikit-models/',
            '--project', PROJECT_ID
        ], check=True)
        
        print(f"✅ Model uploaded to gs://scikit-models/")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model upload failed: {e}")

def get_service_url():
    """Get the current service URL"""
    
    try:
        result = subprocess.run([
            'gcloud', 'run', 'services', 'describe', SERVICE_NAME,
            '--region', REGION,
            '--project', PROJECT_ID,
            '--format', 'value(status.url)'
        ], capture_output=True, text=True, check=True)
        
        service_url = result.stdout.strip()
     
        return service_url
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get service URL: {e}")
        return None

def check_requirements():
    """Check if required tools are installed"""
    tools = ['gcloud', 'docker']
    missing = []
    
    for tool in tools:
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            print(f"✅ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
            print(f"❌ {tool} is not installed")
    
    if missing:
        print(f"\n⚠️  Please install: {', '.join(missing)}")
        return False
    
    # Check if authenticated
    try:
        result = subprocess.run([
            'gcloud', 'config', 'get', 'account'
        ], capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print(f"✅ Authenticated as: {result.stdout.strip()}")
        else:
            print("❌ Not authenticated. Run: gcloud auth login")
            return False
    except subprocess.CalledProcessError:
        print("❌ Not authenticated. Run: gcloud auth login")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "deploy":
            if check_requirements():
                deploy_with_gcloud()
        elif command == "upload-models":
            upload_models_to_storage()
        elif command == "url":
            get_service_url()
        elif command == "check":
            check_requirements()
        else:
            print("Usage: python deploy_api_gcs.py [deploy|upload-models|url|check]")
    else:
        # Default behavior
        if check_requirements():
            deploy_with_gcloud()