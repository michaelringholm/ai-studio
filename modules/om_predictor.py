import os
import keras as ker
import numpy as np
import modules.om_logging as oml

class OMPredictor():
    def __init__(s):
        return
    
    def predict(s,model:ker.Model):
        #np.array([[100.0,105.0,95.0,102.5,1000],  
         #            [107.5, 97.5, 105.0, 100,1000], 
          #           [110.0, 100.0, 107.5, 100,1000]])
        x=np.array([
                    [1.6095628058220817,0,1], # 5
                    [-1.64996921305955,0,1], # 0
                    [-0.020203203618734157,1,0] # 4
        ])
        ai_prediction=model.predict(x=x,batch_size=1)
        oml.debug(f"ai_prediction={ai_prediction}")
        return ai_prediction
    
    def load_model(s,project_folder,project_name)->ker.Model:
        model_architecture_file_path=os.path.join(project_folder,project_name,"model-architecture.json")
        model_weights_file_path=os.path.join(project_folder,project_name,"model-weights.h5")
        with open(model_architecture_file_path) as json_file:
            model_json = json_file.read()
        model:ker.Model=ker.models.model_from_json(model_json)
        model.load_weights(model_weights_file_path)
        return model