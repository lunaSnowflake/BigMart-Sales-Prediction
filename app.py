import numpy as np
import datetime
from flask import Flask, request, jsonify, render_template, Markup
import pickle

app = Flask(__name__)
model = pickle.load(open('Catmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html') # remember to keep the file inside the folder 'templates'

# OUR MODEL STRUCTURE:::::::::::::::::::::::::::::
# Item_Weight                          14.3
# Item_Fat_Content                  Low Fat
# Item_Visibility                    0.0263
# Item_Type                    Frozen Foods
# Item_MRP                          79.4302
# Outlet_Size                          High
# Outlet_Location_Type               Tier 3
# Outlet_Type             Supermarket Type1
# Item_Type_Combined                   Food ;;
# MRP_grp                                 2 ;;

# PREDICTION DATA::::::::::::::::::::::::::::::::::
# *Item_Identifier                          FDW58 xx
# *Item_Weight                              20.75 --
# Item_Fat_Content                        Low Fat --
# *Item_Visibility                       0.007565 --
# Item_Type                           Snack Foods --
# *Item_MRP                              107.8622 --
# *Outlet_Identifier                       OUT049 xx
# Outlet_Establishment_Year                  1999 xx
# Outlet_Size                              Medium --
# Outlet_Location_Type                     Tier 1 --
# Outlet_Type                   Supermarket Type1 --

@app.route('/predict', methods=['POST'])
def predict():
    #* Display text to user
    def web_display(display_text, font_color='white', font_size = 30):
        display_text = Markup(f"<span style='color: {font_color}; font-size: {font_size}px;'>{display_text}</span>")
        return render_template('index.html', display_text=display_text) #display_text is a placeholder in 'index.html'
    
    feature_names = ['Item Identifier', 'Item Weight', 'Item Fat Content', 'Item Visibility', 'Item Type', 'Item MRP', 'Outlet Identifier', \
                        'Outlet Establishment Year', 'Outlet Size', 'Outlet_Location Type', 'Outlet Type']
    
    #* Preprocessing the data
    features = list(request.form.values())
    Item_Identifier = features[0]
    
    #* Typecast
    pred_data = []
    try:
        required_dtypes = [str, float, str, float, str, float, str, int, str, str, str]
        elem_num = -1
        for item, value in zip(required_dtypes, features):
            elem_num += 1
            pred_data.append(item(value))
    except Exception as error:
        display_text = f'Invalid "{feature_names[elem_num]}"'
        return web_display(display_text, font_color='red')

    #* Checking Input Values Validity
    invalid_id = -1
    try:
        if len(pred_data[0]) != 5: #Item Identifier 0
            invalid_id = 0
            raise Exception
        if pred_data[3] > 1: #Item Visibility 3
            invalid_id = 3
            raise Exception
        if pred_data[7] > datetime.datetime.now().year:  #Outlet Establishment Year 7
            invalid_id = 7
            raise Exception
    except:
        display_text = f'Invalid "{feature_names[invalid_id]}"'
        return web_display(display_text, font_color='red')
    
    #* Create Features
    #? Item_Type_Combined
    try:
        AB = Item_Identifier[:2] # 1st 2 char
        map = {'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'}
        pred_data.append(map[AB]) # add new feature
    except Exception as error:
        display_text = f'Invalid "{feature_names[0]}"'
        return web_display(display_text, font_color='red')
    #? MRP_grp
    Item_MRP = int(pred_data[5]) # get MRP
    map = '1' if (Item_MRP < 60) else '2' if (Item_MRP < 140) else '3' if (Item_MRP < 200) else '4'
    pred_data.append(map) # add new feature
    
    #* pop unwanted features
    for i, j in enumerate([0,6,7]):
        pred_data.pop(j-i) # Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year
    
    #* Model Predict
    predicted = np.round(model.predict(pred_data), 2)
    display_text = 'Sales for {}: {}'.format(Item_Identifier, str(predicted))
    return web_display(display_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(debug=True)

