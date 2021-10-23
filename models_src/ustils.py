import tensorflow as tf
BASE_DIR=os.get_cwd()
import json

def write_model(model):
    #try:
    model.save(os.path.join(BASE_DIR,'models_ml','deployment'))
    print("model saved")
    

    #return True
    #except:
        #return False
def read_model():
    # It can be used to reconstruct the model identically.
    reconstructed_model = keras.models.load_model(os.path.join(BASE_DIR,'models_model','deployment'))
    return reconstructed_model

def model_evaluate(model,x,y):
        #Evalutaion of model
    pred = model.predict(x) 
    score = model.evaluate(x, y, verbose=0)
    scores_dict={'Test loss': score[0],'accuracy': score[1]}
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    with open(os.path.join(BASE_DIR,'martrics.json'),"w") as files:
        json.dump(scores_dict,files)