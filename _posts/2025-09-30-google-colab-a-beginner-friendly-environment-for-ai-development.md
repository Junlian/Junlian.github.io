---
layout: post
title: "Google Colab: A Beginner-Friendly Environment for AI Development"
description: "Comprehensive guide to AI agent development and implementation strategies."
date: 2025-09-30
categories: [ai, agent, development, automation]
author: "Junlian"
tags: [ai, agent, development, automation, machine-learning]
seo_title: "Google Colab: A Beginner-Friendly Environment for AI Development - AI Agent Development Guide"
excerpt: "Comprehensive guide to AI agent development and implementation strategies."
---

# Google Colab: A Beginner-Friendly Environment for AI Development

Google Colaboratory (Colab) is a powerful, cloud-based platform that provides an accessible and efficient environment for artificial intelligence (AI) and machine learning (ML) development, particularly for beginners. It eliminates the need for complex local setup by offering a pre-configured, browser-based Jupyter notebook environment with essential Python libraries and free access to computational resources like GPUs and TPUs ([Sarthak, 2023](https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5)). This makes it an ideal starting point for learners and newcomers to AI, enabling them to focus on coding and experimentation rather than installation and configuration hurdles.

Colab supports seamless project organization and execution, allowing users to upload or clone repositories (e.g., from GitHub), mount Google Drive for data storage, and structure code in cells for modular testing and execution ([Sarthak, 2023](https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5)). Its integration with popular AI tools, such as Google’s Gemini LLM, further enhances its utility for modern AI projects ([Sharma, 2025](https://medium.com/@kshitijsharma94/how-to-use-google-colab-with-googles-llm-gemini-a-beginner-s-guide-9d215a6cbd83)). With real-time collaboration features and no-cost access to advanced hardware, Colab democratizes AI development, providing a scalable platform for education and prototyping ([Code With Ebrima, 2025](https://www.udemy.com/course/google-colab-tutorial-2025-from-beginner-basics-to-advance/?srsltid=AfmBOooj3Rtp0hUL3lkDbMHiqAwDn8_Aa2RJZfrHoMduHTbptHBtU0f-)). This introduction explores the setup, environment configuration, and practical workflow of Google Colab, including Python code demonstrations and project structure guidelines tailored for beginners embarking on their AI journey.

## Table of Contents

- Setting Up Google Colab for AI Development
    - Configuring Hardware Acceleration for Optimal Performance
- Check GPU availability
    - Environment Setup and Package Management
- Mount Google Drive for persistent storage
- Install project-specific packages
- Verify installations
- Create and activate virtual environment
- Install exact versions for reproducibility
    - Project Structure and Version Control Integration
- Configure Git
- Clone repository and set up project
- Install package in development mode
- Import custom modules
- Automated commit and push
    - Advanced Session Management and Persistence
- Prevent session timeouts
- Save checkpoint files regularly
- Efficient large file handling
- Compress processed data
- Direct streaming from Drive without full download
    - API Integration and Secure Credential Management
- Store API keys securely
- Retrieve API key from secrets
- Initialize API client securely
- Example API usage with error handling
- config.py in your src directory
- Usage in notebooks
    - Configuring the Development Environment
        - Environment Initialization and Dependency Management
- Add project directory to Python path
- Set environment variables for consistent behavior
    - Advanced Dependency Resolution and Conflict Management
- Check pre-installed versions
- Install with dependency resolution
    - Runtime Configuration and Performance Optimization
- Monitor resource usage
- Configure TensorFlow for optimal performance
    - Development Environment Customization and Extensions
- Install and configure development tools
- Enable useful extensions
- Customize display options
    - Environment Verification and Validation Framework
- Comprehensive environment validation
- Run validation
    - Integrated Development Workflow Configuration
- Set up testing framework integration
- Configure test discovery
- Run tests automatically
    - Structuring and Executing a Python AI Project
        - Modular Code Organization and Import Strategies
- Add project root to Python path
- Import custom modules using absolute imports
    - Data Pipeline Implementation and Management
    - Model Training Orchestration Framework
    - Execution Workflow Automation
- Add to notebook cell
    - Collaborative Development and Code Review Integration
- Usage in team environment





## Setting Up Google Colab for AI Development

### Configuring Hardware Acceleration for Optimal Performance

Google Colab provides complimentary access to GPU and TPU resources, which are essential for accelerating AI model training. To configure hardware acceleration, navigate to **Runtime > Change runtime type** and select either **GPU** or **TPU** from the Hardware accelerator dropdown menu ([Google Colab Beginner's Guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide)). For AI development, the T4 GPU is recommended due to its balance of performance and availability. Note that free-tier users may experience usage limits, with sessions typically disconnecting after 12 hours of continuous activity. The following Python code verifies GPU availability and specifications:

```python
import tensorflow as tf

# Check GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU detected: {gpu_devices[0].name}")
    # Display GPU memory details
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
else:
    print("No GPU available. Switch runtime type to GPU.")
```

Advanced users should note that Colab Pro ($9.99/month) provides priority access to higher-memory GPUs like V100 and A100, which can reduce training time for complex models by up to 40% compared to free-tier resources ([Configuring Google Colab Like A Pro](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573)).

### Environment Setup and Package Management

While Colab comes pre-installed with major AI libraries (TensorFlow 2.15.0, PyTorch 2.2.0, and scikit-learn 1.3.0 as of 2025), project-specific dependencies require careful management. The recommended approach involves creating a `requirements.txt` file in your Google Drive and installing packages systematically:

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Install project-specific packages
!pip install -r /content/drive/MyDrive/your_project/requirements.txt

# Verify installations
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

For reproducible environments, consider using conda-equivalent setups through virtual environments:

```python
# Create and activate virtual environment
!python -m venv /content/colab_env
!source /content/colab_env/bin/activate

# Install exact versions for reproducibility
!pip install tensorflow==2.15.0 torch==2.2.0 transformers==4.35.0
```

This approach ensures version consistency across sessions, critical for debugging and collaboration ([Python Environment Setup Guide](https://www.youtube.com/watch?v=4s7mOZ07tBc)).

### Project Structure and Version Control Integration

A well-organized project structure is fundamental for sustainable AI development. Implement this folder hierarchy in Google Drive:

```
MyDrive/
└── ai_project/
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── external/
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_model_training.ipynb
    │   └── 03_evaluation.ipynb
    ├── src/
    │   ├── data_processing.py
    │   ├── model_architecture.py
    │   └── utils.py
    ├── models/
    ├── requirements.txt
    └── README.md
```

Integrate Git version control through the following setup:

```python
# Configure Git
!git config --global user.email "your_email@example.com"
!git config --global user.name "Your Name"

# Clone repository and set up project
!git clone https://github.com/your_username/your_repo.git /content/drive/MyDrive/ai_project/

# Install package in development mode
!pip install -e /content/drive/MyDrive/ai_project/
```

This structure facilitates modular code development, where the `src` directory contains reusable Python modules that can be imported into Colab notebooks:

```python
# Import custom modules
import sys
sys.path.append('/content/drive/MyDrive/ai_project/src')

from data_processing import DataCleaner
from model_architecture import CustomModel
```

For automated backups, implement this Git push routine:

```python
# Automated commit and push
!cd /content/drive/MyDrive/ai_project && git add .
!cd /content/drive/MyDrive/ai_project && git commit -m "Colab auto-commit"
!cd /content/drive/MyDrive/ai_project && git push origin main
```

### Advanced Session Management and Persistence

Maintaining persistent sessions is crucial for long-running AI training tasks. Implement these strategies to prevent disconnections and manage runtime effectively:

```python
# Prevent session timeouts
from IPython.display import Javascript
def prevent_disconnect():
    display(Javascript('''
    function keepAlive() {
        const req = new XMLHttpRequest();
        req.open("GET", "/api/kernels/" + kernel_id + "/channels", true);
        req.send();
    }
    setInterval(keepAlive, 60000);
    '''))

prevent_disconnect()

# Save checkpoint files regularly
checkpoint_path = "/content/drive/MyDrive/ai_project/models/checkpoint.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)
```

For large datasets, optimize Google Drive integration with these techniques:

```python
# Efficient large file handling
import zipfile
import os

# Compress processed data
with zipfile.ZipFile('/content/drive/MyDrive/ai_project/data/processed.zip', 'w') as zipf:
    for root, dirs, files in os.walk('/content/processed_data'):
        for file in files:
            zipf.write(os.path.join(root, file))

# Direct streaming from Drive without full download
def stream_large_file(file_path):
    import io
    from google.colab import files
    content = files.upload()
    return io.BytesIO(content[0])
```

### API Integration and Secure Credential Management

Secure integration of external APIs like Gemini requires proper credential handling. Implement this secure pattern using Colab's secrets management:

```python
# Store API keys securely
from google.colab import userdata
import os

# Retrieve API key from secrets
gemini_key = userdata.get('GEMINI_API_KEY')
os.environ['GEMINI_API_KEY'] = gemini_key

# Initialize API client securely
from google import genai

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# Example API usage with error handling
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain neural networks briefly"
    )
    print(response.text)
except Exception as e:
    print(f"API Error: {str(e)}")
```

For comprehensive project configuration, create a centralized configuration module:

```python
# config.py in your src directory
import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    DATA_PATH: str = '/content/drive/MyDrive/ai_project/data'
    MODEL_PATH: str = '/content/drive/MyDrive/ai_project/models'
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001

# Usage in notebooks
from src.config import ProjectConfig
config = ProjectConfig()
```

This setup ensures consistent configuration across all project components while maintaining security best practices ([Gemini API Guide](https://medium.com/@kshitijsharma94/how-to-use-google-colab-with-googles-llm-gemini-a-beginner-s-guide-9d215a6cbd83)).


## Configuring the Development Environment

### Environment Initialization and Dependency Management

While previous sections covered hardware acceleration and package installation, this section focuses on environment initialization strategies that ensure consistent behavior across Colab sessions. Google Colab provides a fresh environment for each session, making dependency management crucial for reproducible AI development ([Google Colab FAQ](https://research.google.com/colaboratory/faq.html)).

A critical aspect often overlooked by beginners is the management of Python path and environment variables. The following configuration ensures proper module import paths and environment consistency:

```python
import sys
import os

# Add project directory to Python path
project_path = '/content/drive/MyDrive/ai_project'
if project_path not in sys.path:
    sys.path.append(project_path)

# Set environment variables for consistent behavior
os.environ['PYTHONPATH'] = f"{project_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow verbosity
```

This setup prevents common import errors and ensures that custom modules in the src directory are accessible throughout the notebook session ([Jupyter Documentation](https://jupyter.org/documentation)).

### Advanced Dependency Resolution and Conflict Management

Unlike basic package installation covered previously, advanced dependency management involves resolving version conflicts and ensuring compatibility across the AI stack. Colab's environment comes with pre-installed packages that may conflict with project requirements:

```python
# Check pre-installed versions
!pip show tensorflow numpy pandas

# Install with dependency resolution
!pip install --upgrade --no-deps --force-reinstall tensorflow==2.15.0
!pip check  # Verify no conflicts exist
```

This approach is particularly important when working with specialized AI libraries that have specific version requirements. The `pip check` command helps identify dependency conflicts before they cause runtime errors ([Python Packaging Authority](https://packaging.python.org/)).

### Runtime Configuration and Performance Optimization

Beyond hardware selection, runtime configuration significantly impacts development efficiency. This includes memory management, session persistence, and computational optimization:

```python
# Monitor resource usage
import psutil
def check_system_resources():
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"CPU Usage: {psutil.cpu_percent()}%")

# Configure TensorFlow for optimal performance
import tensorflow as tf
tf.config.optimizer.set_jit(True)  Enable XLA compilation
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
```

These configurations help prevent out-of-memory errors and optimize computational efficiency, particularly important when working with large datasets or complex models ([TensorFlow Performance Guide](https://www.tensorflow.org/guide/performance/overview)).

### Development Environment Customization and Extensions

While Colab provides a standard environment, advanced customization enhances development experience. This includes IDE-like features, custom widgets, and productivity extensions:

```python
# Install and configure development tools
!pip install jupyter_contrib_nbextensions
!jupyter contrib nbextension install --user

# Enable useful extensions
!jupyter nbextension enable codefolding/main
!jupyter nbextension enable execute_time/ExecuteTime

# Customize display options
from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
```

These customizations provide features like code folding, execution timing, and improved layout management, creating a more professional development environment ([Jupyter Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)).

### Environment Verification and Validation Framework

Establishing a robust validation framework ensures environment consistency across sessions and collaborators. This goes beyond simple package checks to comprehensive environment validation:

```python
# Comprehensive environment validation
def validate_environment():
    requirements = {
        'tensorflow': '2.15.0',
        'torch': '2.2.0',
        'transformers': '4.35.0'
    }
    
    issues = []
    for package, expected_version in requirements.items():
        try:
            actual_version = __import__(package).__version__
            if actual_version != expected_version:
                issues.append(f"{package}: expected {expected_version}, got {actual_version}")
        except ImportError:
            issues.append(f"{package}: not installed")
    
    # Check GPU availability
    if not tf.config.list_physical_devices('GPU'):
        issues.append("GPU: not available")
    
    return issues

# Run validation
validation_issues = validate_environment()
if validation_issues:
    print("Environment issues detected:")
    for issue in validation_issues:
        print(f" - {issue}")
else:
    print("Environment validation passed")
```

This validation framework provides immediate feedback on environment consistency, preventing hours of debugging due to environment mismatches ([Software Verification Best Practices](https://medium.com/@ByteWave/harnessing-google-colaboratory-for-ai-programming-a-comprehensive-guide-with-detailed-examples-67e4cec32190)).

### Integrated Development Workflow Configuration

Configuring an integrated workflow that connects Colab with other development tools significantly enhances productivity. This includes setting up automated testing, continuous integration checks, and development tool integration:

```python
# Set up testing framework integration
!pip install pytest pytest-cov

# Configure test discovery
test_config = """
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
"""

with open('/content/drive/MyDrive/ai_project/pyproject.toml', 'a') as f:
    f.write(test_config)

# Run tests automatically
!cd /content/drive/MyDrive/ai_project && python -m pytest -v --cov=src
```

This integrated approach ensures that code developed in Colab meets quality standards and can be seamlessly integrated into larger development workflows ([Python Testing Guide](https://docs.pytest.org/en/stable/)).


## Structuring and Executing a Python AI Project

### Modular Code Organization and Import Strategies

While previous sections addressed project directory structures, this subsection focuses on practical implementation patterns for organizing Python modules and managing imports within Colab's unique environment. Unlike local development environments, Colab requires explicit path configuration to recognize custom modules, making import management a critical consideration for sustainable AI projects ([Google Colab Documentation](https://colab.research.google.com/notebooks/io.ipynb)).

A robust approach involves creating a package structure with `__init__.py` files and using relative imports. Consider this implementation:

```python
# Add project root to Python path
import sys
from pathlib import Path

project_root = Path('/content/drive/MyDrive/ai_project')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import custom modules using absolute imports
from src.data_processing import DataPreprocessor
from src.model_architecture import create_cnn_model
from src.utils import setup_logging, save_artifacts
```

This structure enables clean import patterns while maintaining compatibility with both Colab and local development environments. The key difference from existing content is the focus on import patterns and module organization rather than basic path setup.

### Data Pipeline Implementation and Management

Effective data handling separates successful AI projects from experimental notebooks. This section details structured data pipeline implementation, addressing Colab's ephemeral storage limitations and the need for reproducible data processing ([Medium, 2023](https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5)).

Implement a data pipeline class that handles data versioning, preprocessing, and caching:

```python
class DataPipeline:
    def __init__(self, data_dir, cache_dir='/content/cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_and_preprocess(self, dataset_name, force_refresh=False):
        cache_path = self.cache_dir / f'{dataset_name}_processed.pkl'
        
        if not force_refresh and cache_path.exists():
            return self._load_from_cache(cache_path)
            
        # Implement actual data loading and processing
        raw_data = self._load_raw_data(dataset_name)
        processed_data = self._preprocess_data(raw_data)
        
        self._save_to_cache(processed_data, cache_path)
        return processed_data
```

This approach addresses Colab's temporary storage limitations by implementing smart caching mechanisms that persist across sessions when combined with Google Drive integration.

### Model Training Orchestration Framework

While previous sections covered basic environment setup, this subsection provides a comprehensive framework for orchestrating model training experiments with proper tracking and reproducibility. This includes configuration management, experiment tracking, and model versioning systems tailored for Colab's environment ([Marqo.ai, 2025](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide)).

Implement a training orchestration system:

```python
class TrainingOrchestrator:
    def __init__(self, config_path, experiment_name):
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name
        self.setup_experiment_tracking()
        
    def setup_experiment_tracking(self):
        # Initialize MLflow or similar tracking
        import mlflow
        mlflow.set_tracking_uri("/content/drive/MyDrive/experiments")
        mlflow.set_experiment(self.experiment_name)
        
    def run_experiment(self):
        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(self.config['hyperparameters'])
            
            # Training logic
            model, metrics = self._train_model()
            
            # Log artifacts and model
            mlflow.log_metrics(metrics)
            mlflow.log_artifact('training_artifacts/')
```

This framework ensures that each training run is properly documented and reproducible, addressing a common gap in beginner AI projects where experimentation tracking is often overlooked.

### Execution Workflow Automation

This section focuses on automating the execution workflow from data loading to model deployment, creating a seamless pipeline that can be version-controlled and shared across teams. Unlike previous sections that addressed individual components, this integrates all aspects into a cohesive automated system ([DeepLearning.AI Community, 2023](https://community.deeplearning.ai/t/coding-environment-colab-tips/290232)).

Implement a main execution script that orchestrates the entire workflow:

```python
def main_execution_workflow(config_file='config.yaml'):
    # Initialize components
    data_pipeline = DataPipeline('/content/drive/MyDrive/ai_project/data')
    orchestrator = TrainingOrchestrator(config_file, 'production_experiment')
    
    # Execute pipeline
    try:
        processed_data = data_pipeline.load_and_preprocess('main_dataset')
        orchestrator.run_experiment(processed_data)
        
        # Generate reports and artifacts
        generate_performance_reports()
        save_model_artifacts()
        
        print("Workflow completed successfully")
        
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        raise

# Add to notebook cell
if __name__ == "__main__":
    main_execution_workflow()
```

This automated workflow ensures consistent execution across different runs and facilitates collaboration by providing a standardized approach to project execution.

### Collaborative Development and Code Review Integration

While previous content addressed version control basics, this section expands on collaborative development practices specifically tailored for Colab environments. This includes code review integration, collaborative debugging, and team workflow optimization ([YouTube, 2025](https://www.youtube.com/watch?v=cmNrIcB77D0)).

Implement collaborative development practices:

```python
class CollaborativeDevelopment:
    def __init__(self, project_path, team_members):
        self.project_path = Path(project_path)
        self.team_members = team_members
        self.setup_collaborative_tools()
        
    def setup_collaborative_tools(self):
        # Integrate with collaborative platforms
        self.setup_code_review_system()
        self.configure_shared_environment()
        
    def create_review_notebook(self, issue_number, description):
        """Create a collaborative notebook for specific issues"""
        review_path = self.project_path / f'reviews/issue_{issue_number}.ipynb'
        
        template = {
            'cells': [
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [f'# Code Review: Issue {issue_number}\n\n{description}']
                }
            ]
        }
        
        with open(review_path, 'w') as f:
            json.dump(template, f)
        
        return review_path

# Usage in team environment
collab = CollaborativeDevelopment('/content/drive/MyDrive/ai_project', ['member1@email.com', 'member2@email.com'])
review_notebook = collab.create_review_notebook(42, 'Model architecture optimization')
```

This approach transforms Colab from an individual notebook environment into a collaborative AI development platform, addressing the team coordination aspects that are crucial for real-world AI projects.

## Conclusion

This research demonstrates that Google Colab provides a robust, accessible platform for beginner AI development when configured with specific hardware acceleration, systematic environment management, and structured project organization. The most critical findings indicate that selecting T4 GPU acceleration significantly boosts performance for AI model training, while implementing virtual environments and precise dependency management through `requirements.txt` ensures reproducibility across sessions ([Google Colab Beginner's Guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide)). Furthermore, establishing a modular project structure with clear separation of data, notebooks, source code, and models—integrated with version control—creates a sustainable foundation for AI development that supports both individual learning and collaborative projects ([Medium, 2023](https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5)).

The implications of these findings are substantial for educational and early-stage AI development. Beginners can immediately leverage Colab's free resources while implementing professional-grade development practices, including automated experiment tracking, secure API integration, and collaborative workflows. The integration of MLflow for experiment documentation and smart data caching strategies addresses common pitfalls in reproducibility and resource management ([TensorFlow Performance Guide](https://www.tensorflow.org/guide/performance/overview)). Next steps should involve exploring Colab Pro for enhanced computational resources when projects scale, implementing continuous integration pipelines, and developing more advanced monitoring systems for long-running training jobs ([Google Colab Documentation](https://colab.research.google.com/notebooks/io.ipynb)).


## References

- [https://www.youtube.com/watch?v=Zn5tmx0ynw8](https://www.youtube.com/watch?v=Zn5tmx0ynw8)
- [https://colab.research.google.com/](https://colab.research.google.com/)
- [https://medium.com/@ByteWave/harnessing-google-colaboratory-for-ai-programming-a-comprehensive-guide-with-detailed-examples-67e4cec32190](https://medium.com/@ByteWave/harnessing-google-colaboratory-for-ai-programming-a-comprehensive-guide-with-detailed-examples-67e4cec32190)
- [https://www.youtube.com/watch?v=SuzwJ28mby4](https://www.youtube.com/watch?v=SuzwJ28mby4)
- [https://medium.com/@kshitijsharma94/how-to-use-google-colab-with-googles-llm-gemini-a-beginner-s-guide-9d215a6cbd83](https://medium.com/@kshitijsharma94/how-to-use-google-colab-with-googles-llm-gemini-a-beginner-s-guide-9d215a6cbd83)
- [https://www.youtube.com/watch?v=4s7mOZ07tBc](https://www.youtube.com/watch?v=4s7mOZ07tBc)
- [https://community.deeplearning.ai/t/coding-environment-colab-tips/290232](https://community.deeplearning.ai/t/coding-environment-colab-tips/290232)
- [https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5](https://medium.com/@shibugarg0303/run-entire-python-project-on-google-colab-10de1871c9a5)
- [https://www.youtube.com/watch?v=8KeJZBZGtYo](https://www.youtube.com/watch?v=8KeJZBZGtYo)
- [https://www.geeksforgeeks.org/machine-learning/how-to-run-python-code-on-google-colaboratory/](https://www.geeksforgeeks.org/machine-learning/how-to-run-python-code-on-google-colaboratory/)
- [https://www.youtube.com/watch?v=cmNrIcB77D0](https://www.youtube.com/watch?v=cmNrIcB77D0)
- [https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide)
