# WIP
![Covid-19](Imges/imagexray.png)

# Use Chest X-rays Images in Azure Machine Learning Studio to predict Covid-19 and deploy a web service to Azure Kubernetes Service (AKS)

In this lab, I will walk you through on how to apply machine learning to predict COVID-19 cases from chest X-rays. By following the steps in this lab, you will be able to understand get started. I am hoping to reach individuals who are able to contribute their skills to this effort.

## Deploy Required Resources
Click the below button to upload the provided ARM template to the Azure portal, which is written to automatically deploy and configure the following resources:
  1. An Azure Machine Learning Workspace set to Basic
  2. Azure Key Vault
  3. Application Insights
  4. Storage Account (general purpose v2) with LRS (Locally-redundant storage)
  
 </br>
    <a href="https%3A%2F%2Fraw.githubusercontent.com%2FHrashid789%2Fx-rayPredictionCovid-19%2Fmaster%2FAzureDeploy.json" target="_blank">
        <img src="https://aka.ms/deploytoazurebutton"/>
    </a>

## Dataset

1) [Covid Chestxray dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2) [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2)

### Limitations and Considerations

This is a demo to showcase a Bot Framework, Luis and QnA maker use case.  It is not intended to be a framework or scalable architecture for all scenarios, though it can give you an idea of what your scenario might end up looking like.

## Further reading
- [Microsoft Learn AML with Hands-on-Labs](https://github.com/MicrosoftDocs/mslearn-aml-labs)
- [Azure Machine Learning documentation](https://docs.microsoft.com/en-in/azure/machine-learning/)
- [Deploy a Model in Azure Container Instances](https://docs.microsoft.com/en-in/azure/machine-learning/tutorial-deploy-models-with-aml)

## Deploying a web service to Azure Kubernetes Service (AKS)

### Step 1:- Create an Azure Machine Learning workspace
1.	Sign in to the Azure portal by using the credentials for your Azure subscription.
2.	In the upper-left corner of Azure portal, select + Create a resource.
3.	Use the search bar to find Machine Learning.
4.	Select Machine Learning.
5.	In the Machine Learning pane, select Create to begin.
6.	Provide the required information to configure your new workspace

### Step 2: Create a notebook
After creation of Azure Machine learning workspace, now open the azure ML workspace and go to the “compute” section and create new instances of compute. 
Now launch the studio, and go to the “New” section for creation of the notebook and type the name of the notebook and select the file type as Notebook.
Gave name as “CovidXrayNote.ipynb”

### Step 3: Upload the Data in zip format
Go to the root folder of your user or project, click the upload icon and select the zip file of data for using your project.
Now open your notebook, which we have created already and provide the script for the extraing the zip file into the required path.
 ```bash
Import zipfile
import os
datafile = "CovidXRAY Dataset.zip"
datafile_dbfs = os.path.join("/Data/" + mountname, datafile)

print ("unzipping takes approximatelly 2 minutes")
zip_ref = zipfile.ZipFile(datafile_dbfs, 'r')
zip_ref.extractall("/Data/" + mountname)
zip_ref.close()
 ```
### Step 4: Import the libraries(CovidXrayNote.ipynb)
Import the required libraries.
CovidXrayNote file is used to train and test the model classification of images. And check the accuracy of the model.
### Step 5: Get the workspace(DeployNotebook.ipynb)
This file is used to deploy the model in AKS service.
Import the libraries.
Load existing workspace
```bash
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
ws.write_config()
ws.get_details()
 ```
### Step 6: Register the model
Register an existing trained model, add description and tags.
```bash
## AKS Model Registration 
from azureml.core.model import Model
aksmodel = Model.register(model_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/mlcovidcompute/code/Users/harash/Trained_model', # this points to a local file
                       model_name = "aks_covid.pkl", # this is the name the model is registered as
                       tags = {'area': "covid", 'type': "classification"},
                       description = "covid xray",
                       workspace = ws)

print(aksmodel.name, aksmodel.description, aksmodel.version)
 ```
### Step 7: Create the Environment
Create an environment that the model will be deployed with
```bash
#custom Deployment Environment
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


environment = Environment('covid-environment')
environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'scikit-learn',
    'Pillow',
    'scikit-image==0.15.0',
    'opencv-python==4.2.0.34'
])
 ```
### Step 8: Write the Entry script(score.py)
Write the script that will be used to predict on your model
Now, create a locally “score.py” file and paste the below code and save it.
Go to current project or user folder and upload the score.py file in azure workspace.
```bash
import json
import joblib
from azureml.core.model import Model
import os
import PIL
from PIL import Image
import numpy as np
import pandas as pd
#import cv2
import flask
import pickle
import werkzeug
from werkzeug.utils import secure_filename
from skimage import io
import requests
from io import BytesIO
import azureml.core
from azureml.core import Workspace

# Called when the service is loaded
print('get dir',os.getcwd())

def init():
    global model
    print(os.getcwd())
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('covid-xray-model')
    model = joblib.load(model_path)

    # Called when a request is received
def run(url):
    # Get the input data as a numpy array
    img_array = io.imread(url, as_grey=True)
    img_pil = Image.fromarray(img_array)
    img_300x300 = np.array(img_pil.resize((300, 300), Image.ANTIALIAS))
    img_array = (img_300x300.flatten())
    img_array = img_array.reshape(-1, 1).T

    predictions = model.predict(img_array)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()
 ```
### Step 9: Create the inferenceConfig 
Create the inference config that will be used when deploying the model
```bash
#inferenceConfig
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice

inference_config = InferenceConfig(entry_script='./score.py', environment=environment)
#inf_config = InferenceConfig(entry_script='./score.py', environment=environment)
 ```
### Step 10: AKS Cluster
This is a one time setup. You can reuse this cluster for multiple deployments after it has been created. If you delete the cluster or the resource group that contains it, then you would have to recreate it.
```bash
# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()

aks_name = 'covid-xray-aks1' 
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)
```
### Step 11: Deploy web service to AKS
```bash
## Deploy Script
# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration()

# # Enable token auth and disable (key) auth on the webservice
# aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)

#%%time
aks_service_name ='aks-covid-service1'

aks_service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)


Get the aks service URL and Keys

print(aks_service.scoring_uri)

key1, Key2 = aks_service.get_keys()
print(key1)
```
### Step 12: Test the web service(ClientService.ipynb)
```bash 
import requests

import json
url ='https://qnacovdi19app-bot.azurewebsites.net/image/Normal.jpeg'
url_c ='https://qnacovdi19app-bot.azurewebsites.net/image/Covid.png'

endpoint_aks ='http://52.148.194.176:80/api/v1/service/aks-covid-service1/score'
key1 ='egfG7b8is01WqgDdCvaN0eKohbluGr7y'

# # If (key) auth is enabled, don't forget to add key to the HTTP header.
headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}

# # If token auth is enabled, don't forget to add token to the HTTP header.
# headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + access_token}

resp = requests.post(endpoint_aks, url_c, headers=headers)
res =''
respose =resp.text
respose = respose.replace('[','').replace(']','')
if (respose == '1'):
    res ='Normal'
else:
    res ='Covid'


print("prediction:", res)
```




