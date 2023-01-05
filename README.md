# Antimony Meter App

## Install 

1. Prerequisite: python3, pip, virtualenv  
[Here's](https://sourabhbajaj.com/mac-setup/Python/) a tutorial on installing these on MacOS  

2. Clone the repository to your local machine  
`git clone https://github.com/bensonjing/chem-app-api.git`

3. Change into the project directory  
`cd chem-app-api/`  

4. Install packages  
`pip install -r requirements.txt`  

5. Start local server  
`gunicorn app:app`  


## Usage  

- Get concentration of certain image:  
Access `https://chem-app-api.herokuapp.com/pic` using POST method with target image 

- Update model by adding more training images:
    1. Add images into`photo/` folder  
    2. Update `info.txt` by writing down the concentration and filename respectively
    3. Open up [client](https://github.com/bensonjing/chem-app-client) and upload image once  
