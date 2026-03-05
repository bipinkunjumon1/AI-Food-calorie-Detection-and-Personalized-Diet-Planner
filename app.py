from flask import Flask, redirect, url_for, render_template, request, flash, session
import re
import os
import sqlite3
import hashlib
from base64 import decode
import requests
import json
from nutritionix import Nutritionix
from decimal import Decimal
from datetime import date, datetime,timedelta 
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder 
import random
from flask import Flask, render_template, request
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np

# Indian Detection Model (your existing detection best.pt)
indian_model_path = "best.pt"

# Western Classification Model
western_model = YOLO("runs/classify/train3/weights/best.pt")


app = Flask(__name__)
app.secret_key='honsproject'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.globals.update(zip=zip)

bf_meal = []
onefood = []
lunch_meal = []
dinner_meal = []
snack_meal = []

# app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class_mapping = {
    0: {'label': 'aloo-gobi', 'calories': 108},
    1: {'label': 'aloo-fry', 'calories': 125},
    2: {'label': 'dum-aloo', 'calories': 164},
    3: {'label': 'fish-curry', 'calories': 241},
    4: {'label': 'ghevar', 'calories': 61},
    5: {'label': 'green-chutney', 'calories': 21},
    6: {'label': 'gulab-jamun', 'calories': 145},
    7: {'label': 'idli', 'calories': 40},
    8: {'label': 'jalebi', 'calories': 150},
    9: {'label': 'chicken-seekh-kebab', 'calories': 158},
    10: {'label': 'kheer', 'calories': 266},
    11: {'label': 'kulfi', 'calories': 136},
    12: {'label': 'bhature', 'calories': 230}, 
    13: {'label': 'lassi', 'calories': 183},
    14: {'label': 'mutton-curry', 'calories': 298},
    15: {'label': 'onion-pakoda', 'calories': 80},
    16: {'label': 'palak-paneer', 'calories': 338},
    17: {'label': 'poha', 'calories': 270},
    18: {'label': 'rajma-curry', 'calories': 235},
    19: {'label': 'rasmalai', 'calories': 188},
    20: {'label': 'samosa', 'calories': 308},
    21: {'label': 'shahi-paneer', 'calories': 261},
    22: {'label': 'white-rice', 'calories': 135},
    23: {'label': 'bhindi-masala', 'calories': 225},
    24: {'label': 'chicken-biryani', 'calories': 348},
    25: {'label': 'chai', 'calories': 54},
    26: {'label': 'chole', 'calories': 311},
    27: {'label': 'coconut-chutney', 'calories': 105},
    28: {'label': 'dal-tadka', 'calories': 260},
    29: {'label': 'dosa', 'calories': 106}
}

yolo_food_calories = {
    "pizza": 266,
    "burger": 295,
    "hot_dog": 150,
    "french_fries": 312,
    "steak": 271,
    "pancake": 227,
    "ice_cream": 207,
    "fried_rice": 238,
    "donut": 452,
    "cup_cake": 305
}

def predict_food_yolo(image_path):
    results = yolo_model.predict(image_path, verbose=False)
    cls_id = results[0].probs.top1
    food_name = results[0].names[cls_id]
    confidence = float(results[0].probs.top1conf)
    calories = yolo_food_calories.get(food_name, "Unknown")
    return food_name, confidence, calories

# ========== SMART AUTO DETECT FUNCTION ========== #

def smart_food_predict(img):
    # ---------- Indian Detection ----------
    indian_bytes, indian_total_cal, indian_items = detect_and_visualize(
        img, indian_model_path, class_mapping
    )

    indian_conf_score = len(indian_items)  # number of detected items


    # ---------- Western Classification ----------
    temp_path = "temp_auto.jpg"
    cv2.imwrite(temp_path, img)

    western_results = western_model.predict(temp_path, verbose=False)
    western_conf = float(western_results[0].probs.top1conf)
    western_class_id = western_results[0].probs.top1
    western_food = western_results[0].names[western_class_id]
    western_cal = yolo_food_calories.get(western_food, 0)

    # ---------- Decision Logic ----------
    if indian_conf_score > 0:
        return "Indian", indian_bytes, indian_total_cal, indian_items
    elif western_conf > 0.60:   # threshold to avoid wrong prediction
        return "Western", western_food, western_cal, western_conf
    else:
        return "Unknown", None, None, None


def calculate_total_calories(class_label, count):
    class_info = class_mapping.get(class_label, {'label': 'unknown', 'calories': 0})
    calories_per_item = class_info['calories']
    total_calories = count * calories_per_item
    return total_calories

def detect_and_visualize(img, model_path, class_mapping, confidence_threshold=0.25):
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    # RGB FIX
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    model = YOLO(model_path)
    results = model.predict(source=img, conf=confidence_threshold)

    detected_items = {}   # ✅ dictionary now

    total_calories = 0
    resized_img = cv2.resize(img, (800, 400))

    scaling_factor_x = 800 / img.shape[1]
    scaling_factor_y = 400 / img.shape[0]

    detections = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()
    classes = results[0].boxes.cls.tolist()

    for i in range(len(detections)):
        box = [int(v) for v in detections[i]]
        resized_box = [
            int(box[0] * scaling_factor_x),
            int(box[1] * scaling_factor_y),
            int(box[2] * scaling_factor_x),
            int(box[3] * scaling_factor_y)
        ]

        class_index = int(classes[i])
        conf = confidences[i]

        if conf > 0.4:
            class_info = class_mapping.get(class_index, {'label': 'unknown', 'calories': 0})
            class_label = class_info['label']
            calories = class_info['calories']
            detected_items[class_label] = detected_items.get(class_label, 0) + 1
            # ✅ correct total calorie calculation
            total_calories += calories


            cv2.putText(
                resized_img,
                f'{class_label} ({calories} kcal) {conf:.2f}',
                (resized_box[0], resized_box[1]),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                2
            )
            cv2.rectangle(
                resized_img,
                (resized_box[0], resized_box[1]),
                (resized_box[2], resized_box[3]),
                (255, 0, 255),
                2
            )

    _, result_image = cv2.imencode('.jpg', resized_img)
    result_bytes = result_image.tobytes()

    return result_bytes, total_calories, detected_items


    items_with_calories = []
    for i in range(30):
        if(detected_items[i] != 0):
            item_cal = class_mapping[i].get('calories') * detected_items[i]
            items_with_calories.append({'label': class_mapping[i].get('label'), 'calories': f"{detected_items[i]} * {class_mapping[i].get('calories')}.00 = {item_cal}", 'count': detected_items[i]})
    return result_bytes, total_calories, items_with_calories




@app.route('/')
@app.route('/first')
def first():
    return render_template('home.html')


@app.route('/predict')
def index1():
    return render_template('index1.html')

@app.route('/about1')
def about1():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index1.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index1.html', error="")

    if file and allowed_file(file.filename):
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        
        # Change this line:
        result_bytes, total_calories, items_with_calories = detect_and_visualize(img, "best.pt", class_mapping)
        
        return render_template('index1.html', filename=f'data:image/jpg;base64,{base64.b64encode(result_bytes).decode()}', total_calories=total_calories, items_with_calories=items_with_calories, name=file.filename)


@app.route("/delete_food", methods =['GET','POST'])
def delete_food():
    if request.method == 'GET':
        return redirect(url_for('add_successful'))

    else:
        mealtime = request.form['mealtime']
        item = request.form['fname']

        uid = session['uid']
        track_date = str(datetime.today().strftime ('%Y-%m-%d'))
        
        bf_list = []
        lunch_list = []
        dinner_list = []
        snack_list = []
        # for i in session['bf_numbers']:

            
        # 	print(float(i))
        # 	session.modified = True
        # 	print(type(i))
        if mealtime == "Breakfast":
            
            for i in bf_meal:
                if item == i[0]:
                    bf_meal.remove(i)

            for i in session['bf_meal']:
                if item == i[0]:
                    session['bf_meal'].remove(i)
                    session.modified = True

            
                    

                
                    for j in session['bf_numbers']:
                        j = round(Decimal(j), 2)
                        bf_list.append(j)
                    
                
                    bf_list[0]-=round(Decimal(i[3]), 2)
                    bf_list[1]-=round(Decimal(i[4]), 2)
                    bf_list[2]-=round(Decimal(i[5]), 2)
                    bf_list[3]-=round(Decimal(i[6]), 2)

                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                        
                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            track_info= cur.fetchall()
                            if not track_info:
    # Create row for today if it doesn't exist
                                cur.execute("""
                                    INSERT INTO tracking
                                    (track_date, u_id, track_breakfast, track_lunch,
                                    track_snack, track_dinner,
                                    track_calorie, track_protein, track_carb, track_fat)
                                    VALUES (?, ?, '', '', '', '', 0, 0, 0, 0)
                                    """, (track_date, uid))
                                conn.commit()

    # Fetch again
                                cur.execute("select * from tracking where track_date=? and u_id=?",
                                    (track_date, uid,))
                                track_info = cur.fetchall()
                
                            calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
                            protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
                            carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
                            fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
                        
                            cur2 = conn.cursor()
                            cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            u_data = cur2.fetchone()

                            item_in_db = u_data[2].split(",")[:-1]
                            item_to_db = ""

                            for i in item_in_db:
                                if item == i:
                                    item_in_db.remove(i)

                            item_to_db = ",".join(item_in_db)+","

                            cur2.execute("update tracking set track_breakfast=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                            conn.commit()

                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()


                    session['bf_numbers'].clear()

                    [x for x in session['bf_numbers'] if x]

                    session.modified = True
                    for i in bf_list:
                        session['bf_numbers'].append(i)	
                        session.modified = True

                    

                    
                    
                    

        
        if mealtime == "Lunch":
            foodlist = session['lunch_meal']

            for i in lunch_meal:
                if item == i[0]:
                    lunch_meal.remove(i)

            for i in foodlist:
                if item == i[0]:
                    foodlist.remove(i)
                
            
                    for j in session['lunch_numbers']:
                        j = round(Decimal(j), 2)
                        lunch_list.append(j)
                    
                
                    lunch_list[0]-=round(Decimal(i[3]), 2)
                    lunch_list[1]-=round(Decimal(i[4]), 2)
                    lunch_list[2]-=round(Decimal(i[5]), 2)
                    lunch_list[3]-=round(Decimal(i[6]), 2)

                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                        
                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            track_info= cur.fetchall()
                
                            calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
                            protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
                            carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
                            fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
                        
                            cur2 = conn.cursor()
                            cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            u_data = cur2.fetchone()

                            item_in_db = u_data[3].split(",")[:-1]
                            item_to_db = ""

                            for i in item_in_db:
                                if item == i:
                                    item_in_db.remove(i)

                            item_to_db = ",".join(item_in_db)+","
                            
                            cur2.execute("update tracking set track_lunch=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                            conn.commit()

                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()

                    session['lunch_numbers'].clear()

                    [x for x in session['lunch_numbers'] if x]

                    session.modified = True
                    for i in lunch_list:
                        session['lunch_numbers'].append(i)	
                        session.modified = True


        if mealtime == "Snack":
            foodlist = session['snack_meal']

            for i in snack_meal:
                if item == i[0]:
                    snack_meal.remove(i)

            for i in foodlist:
                if item == i[0]:
                    foodlist.remove(i)
                
            
                    for j in session['snack_numbers']:
                        j = round(Decimal(j), 2)
                        snack_list.append(j)
                    
                
                    snack_list[0]-=round(Decimal(i[3]), 2)
                    snack_list[1]-=round(Decimal(i[4]), 2)
                    snack_list[2]-=round(Decimal(i[5]), 2)
                    snack_list[3]-=round(Decimal(i[6]), 2)

                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                        
                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            track_info= cur.fetchall()
                
                            calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
                            protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
                            carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
                            fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
                        
                            cur2 = conn.cursor()
                            cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            u_data = cur2.fetchone()

                            item_in_db = u_data[4].split(",")[:-1]
                            item_to_db = ""

                            for i in item_in_db:
                                if item == i:
                                    item_in_db.remove(i)

                            item_to_db = ",".join(item_in_db)+","
                
                            cur2.execute("update tracking set track_snack=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                            conn.commit()

                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()

                    session['snack_numbers'].clear()

                    [x for x in session['snack_numbers'] if x]

                    session.modified = True
                    for i in snack_list:
                        session['snack_numbers'].append(i)	
                        session.modified = True

                    

        if mealtime == "Dinner":
            foodlist = session['dinner_meal']
            for i in dinner_meal:
                if item == i[0]:
                    dinner_meal.remove(i)

            for i in foodlist:
                if item == i[0]:
                    foodlist.remove(i)
                
            
                    for j in session['dinner_numbers']:
                        j = round(Decimal(j), 2)
                        dinner_list.append(j)
                    
                
                    dinner_list[0]-=round(Decimal(i[3]), 2)
                    dinner_list[1]-=round(Decimal(i[4]), 2)
                    dinner_list[2]-=round(Decimal(i[5]), 2)
                    dinner_list[3]-=round(Decimal(i[6]), 2)

                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                        
                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            track_info= cur.fetchall()
                
                            calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
                            protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
                            carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
                            fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
                        
                            cur2 = conn.cursor()
                            cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                            u_data = cur2.fetchone()

                            item_in_db = u_data[5].split(",")[:-1]
                            item_to_db = ""

                            for i in item_in_db:
                                if item == i:
                                    item_in_db.remove(i)

                            item_to_db = ",".join(item_in_db)+","
                            cur2.execute("update tracking set track_dinner=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                            conn.commit()

                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()

                    session['dinner_numbers'].clear()

                    [x for x in session['dinner_numbers'] if x]

                    session.modified = True
                    for i in dinner_list:
                        session['dinner_numbers'].append(i)	
                        session.modified = True

                    
                
                                
        return render_template('add_food.html',mealtime=mealtime)
    


def get_connection():
    conn = sqlite3.connect('diet_recommendation.db')
    conn.row_factory=sqlite3.Row # to be able to reference by column name
    return conn

def pwd_security(passwd):
    """A strong password must be at least 8 characters long
       and must contain a lower case letter, an upper case letter,
       and at least 3 digits.
       Returns True if passwd meets these criteria, otherwise returns False.
       """
    # check password length
    # check password for uppercase, lowercase and numeric chars
    hasupper = False	
    haslower = False
    digitcount = 0
    digit= False
    strong = False
    length = True
    special = False
    for c in passwd:
        if (c.isupper()==True):
            hasupper= True
        elif (c.islower()==True):
            haslower=True
        elif (c.isdigit()==True):
            digitcount+=1
            digit = True
        elif re.findall('[^A-Za-z0-9]',c):
            special = True
    if hasupper == True and haslower == True and digit == True and special == True:
        strong = True
    if len(passwd) <8:
        length = False
    return strong,haslower,hasupper,digit,length, special

def pwd_encode(pwd):
    secure_pwd =hashlib.md5(pwd.encode()).hexdigest()
    return secure_pwd





@app.route("/update_profile", methods =['GET','POST'] )
def edit_weight():
    if request.method == 'GET':

        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from user where u_id=?",(session['uid'],))
                u_data = cur.fetchone()
                name = u_data[1]
                age = u_data[4]
                weight = u_data[6]
                email = u_data[5]
                password = session['u_pass']
                gender = u_data[3]
                ft = u_data[7]
                inch = u_data[8]
                vegan = u_data[12]
                allergy = u_data[13]

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        

        return render_template('edit_profile.html', name= name,
                                                    age = age,
                                                    weight = weight,
                                                    email = email,
                                                    password = password,
                                                    gender = gender,
                                                    ft = ft,
                                                    inch = inch,
                                                    vegan = vegan,
                                                    allergy = allergy)
    else:
        # name = u_data[1]
        # age = u_data[4]
        # weight = u_data[6]
        # email = u_data[5]
        # password = session['u_pass']
        # gender = u_data[3]
        # ft = u_data[7]
        # inch = u_data[8]
        # vegan = u_data[12]
        # allergy = u_data[13]

        return redirect(url_for('profile'))

@app.route("/add_food", methods =['GET','POST'] )
def add_food():
    if request.method == 'GET':
        
        return render_template('add_food.html')

    else:
        mealtime = request.form['mealtime']
        
        
        return render_template('add_food.html',mealtime=mealtime)

def food_list(meal,item):
    
    meal.append(item)

    return meal

def one_food(item,portion,p_type):
    
    meal = [item,portion,p_type]

    return meal

@app.route("/add_successful", methods =['GET','POST'] )
def add_successful():
    uid = session['uid']
    track_date = datetime.today().strftime('%Y-%m-%d')
    food = None
    track_info = None

    if request.method == 'GET':
        return render_template('add_food.html')

    else:
        mealtime = request.form['mealtime']
        item_name = request.form['food']
        foodPortion = int(request.form['portion'])
        portion_type = request.form['portion_type']

        # --- CALORIENINJAS API SETUP ---
        api_key = "mhkTZiqa39KIhO01tDyeKw==m5mAjgzs4k8lxCDN" # Paste your key here
        query = f"{foodPortion} {portion_type} {item_name}"
        url = f"https://api.calorieninjas.com/v1/nutrition?query={query}"

        # CalorieNinjas uses a GET request
        response = requests.get(url, headers={'X-Api-Key': api_key})

        if response.status_code == requests.codes.ok:
            data = response.json()
            
            # Check if any food was found
            if not data['items']:
                 return "Food item not recognized. Try a simpler name."

            item = data['items'][0] 

            # Map CalorieNinjas data to your existing variables
            finalCalorie = item.get('calories')
            finalProtein = item.get('protein_g')
            finalCarb = item.get('carbohydrates_total_g')
            finalFat = item.get('fat_total_g')
            unit = portion_type # Using your form's unit since API response varies

            # This maintains your exact list structure for the rest of the code
            food = [item_name, foodPortion, unit, finalCalorie, finalProtein, finalCarb, finalFat]
            
            try:
                with get_connection() as conn:
                    cur = conn.cursor()
                    
                    # Ensure record exists for today
                    cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                    track_info = cur.fetchall()
                    
                    if not track_info:
                        cur.execute("""INSERT INTO tracking (track_date, u_id, track_breakfast, track_lunch, track_snack, track_dinner, track_calorie, track_protein, track_carb, track_fat) 
                                       VALUES (?, ?, '', '', '', '', 0, 0, 0, 0)""", (track_date, uid))
                        conn.commit()
                        cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
                        track_info = cur.fetchall()

                    # Your original logic for calculating totals
                    calorie = round(Decimal(track_info[0][6]), 2) + round(Decimal(food[3]), 2)
                    protein = round(Decimal(track_info[0][7]), 2) + round(Decimal(food[4]), 2)
                    carb = round(Decimal(track_info[0][8]), 2) + round(Decimal(food[5]), 2)
                    fat = round(Decimal(track_info[0][9]), 2) + round(Decimal(food[6]), 2)
                    
                    cur2 = conn.cursor()
                    
                    # Update DB based on mealtime (Your exact DB logic)
                    if mealtime == "Breakfast":
                        meal_input = track_info[0][2] + food[0] +","
                        cur2.execute("update tracking set track_breakfast=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                    
                    elif mealtime == "Lunch":
                        meal_input = track_info[0][3] + food[0] +","
                        cur2.execute("update tracking set track_lunch=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))

                    elif mealtime == "Snack":
                        meal_input = track_info[0][4] + food[0] +","
                        cur2.execute("update tracking set track_snack=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))

                    elif mealtime == "Dinner":
                        meal_input = track_info[0][5] + food[0] +","
                        cur2.execute("update tracking set track_dinner=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
                    
                    conn.commit()

            except sqlite3.Error as e:
                return (f'Database Error: {e}')

        else:
            # Replaced the Nutritionix error message with a generic one
            return f"Error connecting to nutrition database (Status: {response.status_code})"

        # --- ALL SESSION UPDATES REMAIN UNCHANGED ---
        if mealtime == "Breakfast":
            session['bf_numbers'] = [0,0,0,0]
            session['bf_meal'] = food_list(bf_meal,food)
            for i in session['bf_meal']:
                session['bf_numbers'][0]+=round(Decimal(i[3]), 2)
                session['bf_numbers'][1]+=round(Decimal(i[4]), 2)
                session['bf_numbers'][2]+=round(Decimal(i[5]), 2)
                session['bf_numbers'][3]+=round(Decimal(i[6]), 2)

        elif mealtime == "Lunch":
            session['lunch_numbers'] = [0,0,0,0]
            session['lunch_meal'] = food_list(lunch_meal,food)
            for i in session['lunch_meal']:            
                session['lunch_numbers'][0]+=round(Decimal(i[3]), 2)
                session['lunch_numbers'][1]+=round(Decimal(i[4]), 2)
                session['lunch_numbers'][2]+=round(Decimal(i[5]), 2)
                session['lunch_numbers'][3]+=round(Decimal(i[6]), 2)
        
        elif mealtime == "Snack":
            session['snack_numbers'] = [0,0,0,0]
            session['snack_meal'] = food_list(snack_meal,food)
            for i in session['snack_meal']:
                session['snack_numbers'][0]+=round(Decimal(i[3]), 2)
                session['snack_numbers'][1]+=round(Decimal(i[4]), 2)
                session['snack_numbers'][2]+=round(Decimal(i[5]), 2)
                session['snack_numbers'][3]+=round(Decimal(i[6]), 2)

        elif mealtime == "Dinner":
            session['dinner_numbers'] = [0,0,0,0]
            session['dinner_meal'] = food_list(dinner_meal,food)
            for i in session['dinner_meal']:
                session['dinner_numbers'][0]+=round(Decimal(i[3]), 2)
                session['dinner_numbers'][1]+=round(Decimal(i[4]), 2)
                session['dinner_numbers'][2]+=round(Decimal(i[5]), 2)
                session['dinner_numbers'][3]+=round(Decimal(i[6]), 2)

        return render_template('add_food.html',mealtime=mealtime)

@app.route("/track", methods = ['GET','POST'])
def track():
    if request.method == 'GET':
        uid = session['uid']
        
        track_date = str(datetime.today().strftime ('%Y-%m-%d'))

        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                track_info= cur.fetchall()


                if track_info:
                    pass
                else:
                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                            breakfast=""
                            lunch = ""
                            dinner = ""
                            snack = ""
                            cur.execute("insert into tracking (track_date,u_id,track_calorie,track_protein,track_fat,track_carb,track_breakfast,track_lunch,track_snack,track_dinner) values (?,?,0,0,0,0,?,?,?,?)",(track_date,uid,breakfast,lunch,snack,dinner))
                            conn.commit()

                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                            track_info= cur.fetchall()
                            if track_info[0][2] == ",":
                                cur.execute("update tracking set track_breakfast=?",("",))
                                conn.commit()
                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()

                cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                track_info= cur.fetchall()

                protein_goal = session['u_info'][14]
                carb_goal = session['u_info'][15]
                fat_goal = session['u_info'][16]
                breakfast = track_info[0][2]
                lunch = track_info[0][3]
                snack = track_info[0][4]
                dinner = track_info[0][5]

                    
                protein_consumed = track_info[0][7]
                carb_consumed = track_info[0][8]
                fat_consumed = track_info[0][9]

                calorie_goal = session['u_info'][17]
                calorie_consumed = track_info[0][6]

                protein_percent = "{:.2f}".format((protein_consumed/protein_goal) * 100)
                carb_percent = "{:.2f}".format((carb_consumed/carb_goal) * 100)
                fat_percent = "{:.2f}".format((fat_consumed/fat_goal) * 100)
                calorie_percent = "{:.2f}".format((calorie_consumed/calorie_goal) * 100)
                    
                try:
                    with get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("update tracking set track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (calorie_consumed,protein_consumed,carb_consumed,fat_consumed,track_date,uid,))
                        conn.commit()

                except sqlite3.Error as e:
                    return (f'{e}')
                finally:
                    conn.close()
                    
                return render_template('track.html',p_goal=protein_goal,
                                            c_goal = carb_goal,
                                            f_goal=fat_goal,
                                            p_consumed = protein_consumed,
                                            c_consumed=carb_consumed,
                                            f_consumed = fat_consumed,
                                            p_percent = protein_percent,
                                            c_percent = carb_percent,
                                            f_percent = fat_percent,
                                            cal_percent = calorie_percent,
                                            cal_goal = calorie_goal,
                                            cal_consumed = calorie_consumed,
                                            breakfast = breakfast,
                                            lunch = lunch,
                                            snack = snack,
                                            dinner = dinner,
                                            )
                        
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        
    else:
        return render_template('track.html')


@app.route("/register", methods = ['GET','POST'])
def citizen_register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form['name']
        
        email = request.form['email']
        password = request.form['password']
        
        return render_template('profilesetup.html',name=name,email=email,password=password)

@app.route("/login", methods = ['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        
        session['uid'] = 0
        email = request.form['email']
        password = request.form['password']
        secure_pwd = pwd_encode(password)
        msg=''
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from user where u_email=?",(email,))
                u_info= cur.fetchall()
                if not u_info:
                    flash(f'The email address ({email}) that you entered does not exist in our database.')
                    return redirect(url_for('login'))
                else:
                    for row in u_info:
                        session['uid'] = row[0]
                        u_pass = row[2] 
                        u_name = row[1]
                        u_date = row[-1]
                    
                    if secure_pwd == u_pass:
                        days = []
                        flash(f'Your have successfully logged in as {u_name}')
                        session['u_logged'] = True
                        session['u_info'] = []
                        session['u_pass'] = password 

                        track_date = datetime.today().strftime ('%Y-%m-%d')
                        sdate = datetime.strptime(u_date, '%Y-%m-%d').date()
                        edate = datetime.strptime(track_date, '%Y-%m-%d').date()
                        delta = edate - sdate     

                        for i in range(delta.days + 1):
                            day = sdate + timedelta(days=i)
                            days.append(str(day))
                            journey = len(days)

                        try:
                            with get_connection() as conn:
                                cur = conn.cursor()
                                cur2 = conn.cursor()
                                cur2.execute("update user set u_journey=? where u_id=?", (journey,session['uid'],))
                                conn.commit()

                                cur.execute("select * from user where u_id=?",(session['uid'],))
                                u_info = cur.fetchone()
                
                                for row in u_info:
                                    session['u_info'].append(row)

                        except sqlite3.Error as e:
                            return (f'{e}')
                        finally:
                            conn.close()

                        return redirect(url_for('index'))
                    else:
                        session.pop('uid',None)
                        flash('Sorry the credentails you are using are invalid')
                        return redirect(url_for('login'))

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

@app.route("/setup", methods = ['GET','POST'])
def profilesetup():
    if request.method == 'GET':
        return render_template('profilesetup.html')

    else:
        name = request.form['name']
        email = request.form['email']
        passwd = request.form['password']
        password = pwd_encode(passwd)
        age = int(request.form['age'])
        gender = request.form['gender']
        vegan = request.form['vegan']
        allergy = request.form['allergy']
        
        # 1. Capture weight as KG
        weight_kg = float(request.form['weight']) 
        feet = int(request.form['feet'])
        inches = int(request.form['inches'])
        activity_level = request.form['activity']

        # 2. Convert Height to Centimeters (Metric Requirement)
        # 1 foot = 30.48 cm, 1 inch = 2.54 cm
        height_cm = (feet * 30.48) + (inches * 2.54)
        
        bmr = 0
        body_status = ""
        
        # 3. Metric BMI Formula: weight (kg) / [height (m)]^2
        height_m = height_cm / 100
        BMI = weight_kg / (height_m * height_m)
        bodyfat = 0

        # 4. Metric Mifflin-St Jeor Equation
        if gender == "male":
            # Formula: (10 * weight) + (6.25 * height) - (5 * age) + 5
            bmr = int((10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5)
            bodyfat = int((1.20 * BMI) + (0.23 * age) - 16.2)
        else:
            # Formula: (10 * weight) + (6.25 * height) - (5 * age) - 161
            bmr = int((10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161)
            bodyfat = int((1.20 * BMI) + (0.23 * age) - 5.4)

        calorie = 0

        # Activity Multipliers remain the same
        if activity_level == "sedentary":
            calorie = int(bmr * 1.2)
        elif activity_level == "lightly active":
            calorie = int(bmr * 1.375)
        elif activity_level == "moderately active":
            calorie = int(bmr * 1.55)
        elif activity_level == "very active":
            calorie = int(bmr * 1.725)
        elif activity_level == "extra active":
            calorie = int(bmr * 1.9)

        # BMI Status Logic remains the same
        if BMI < 18.5:
            body_status = "underweight"
        elif 18.5 <= BMI <= 24.9:
            body_status = "healthy weight"
        elif 25 <= BMI <= 29.9:
            body_status = "overweight"
        elif BMI >= 30:
            body_status = "obese"

        # Macro calculations (based on your existing 500 calorie deficit logic)
        protein = int(((calorie - 500) * 0.30) / 4)
        carb = int(((calorie - 500) * 0.40) / 4)
        fat = int(((calorie - 500) * 0.30) / 9)
        fiber = int(calorie / 1000 * 14)
        
        journey = 1

        try:
            with get_connection() as conn:
                db = conn.cursor()
                # Ensure weight_kg is passed to the u_weight column
                db.execute("""
                    INSERT INTO user (
                        u_username, u_email, u_password, u_age, u_gender, 
                        u_vegan, u_allergy, u_weight, u_feet, u_inches, 
                        u_bmi, u_activitylevel, u_protein, u_carb, u_fat, 
                        u_fiber, u_calories, u_journey, u_bodyfat, u_status, u_startdate
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    name, email, password, age, gender, 
                    vegan, allergy, weight_kg, feet, inches, 
                    int(BMI), activity_level, protein, carb, fat, 
                    fiber, calorie, journey, bodyfat, body_status, 
                    datetime.today().strftime('%Y-%m-%d')
                ))
                conn.commit()
                flash('Successfully Registered')

        except sqlite3.Error as e:
            return (f'Database Error: {e}')
        finally:
            conn.close()

        return redirect(url_for('login'))

@app.route("/profile", methods=['GET', 'POST'])
def profile():
    # 1. Check if user is logged in
    if 'uid' not in session:
        flash("Please log in to view your profile.", "warning")
        return redirect(url_for('login'))  # change 'login' to your actual login route

    uid = session['uid']

    if request.method == 'GET':
        try:
            with get_connection() as conn:
                conn.row_factory = sqlite3.Row  # optional: makes row dict-like
                db = conn.cursor()
                db.execute("SELECT * FROM user WHERE u_id = ?", (uid,))
                user = db.fetchone()

                if user is None:
                    flash("User profile not found.", "danger")
                    return redirect(url_for('home'))  # or logout, etc.

                # Convert to dict if you prefer (optional)
                user_dict = dict(user) if isinstance(user, sqlite3.Row) else user

                return render_template('profile.html', u_info=user_dict)

        except sqlite3.Error as e:
            flash(f"Database error: {str(e)}", "danger")
            return render_template('profile.html', u_info=None)  # or redirect

    else:  # POST - handle profile update
        # Example: update name, weight, height, etc.
        # Add your actual form fields here
        try:
            name = request.form.get('name', '').strip()
            weight = request.form.get('weight')
            height = request.form.get('height')
            age = request.form.get('age')
            # ... other fields ...

            # Basic validation
            if not name:
                flash("Name is required.", "warning")
                return redirect(url_for('profile'))

            with get_connection() as conn:
                db = conn.cursor()
                db.execute("""
                    UPDATE user 
                    SET name = ?, weight = ?, height = ?, age = ?
                    WHERE u_id = ?
                """, (name, weight, height, age, uid))
                
                conn.commit()

            flash("Profile updated successfully!", "success")
            return redirect(url_for('profile'))

        except sqlite3.Error as e:
            flash(f"Failed to update profile: {str(e)}", "danger")
            return redirect(url_for('profile'))
        except ValueError as e:
            flash(f"Invalid input: {str(e)}", "warning")
            return redirect(url_for('profile'))


@app.route("/recommendation", methods = ['GET','POST'])
def recommendation():
    if request.method == 'GET':
        return render_template('recommendation.html')

    else:
        dataset = pd.read_csv('dietdataset.csv')

        dataset = pd.DataFrame(data=dataset.iloc[:,0:10].values,columns = ['meal_name','carb','meat','vege','fruit', 'type','vegan','allergy','time'])
        le = LabelEncoder()
        dataset_encoded = dataset.iloc[:,0:10]
        for i in dataset_encoded:
            dataset_encoded[i] = le.fit_transform(dataset_encoded[i])
            
            model = pickle.load(open('model1','rb'))

        bf_vege_input = []
        bf_meat_input = []
        bf_carb_input = []
        bf_fruit_input = []

        bf_vege = random.choice(request.form.getlist('vege'))
        bf_meat = random.choice(request.form.getlist('meat'))
        bf_carb = random.choice(request.form.getlist('carb'))
        bf_fruit = random.choice(request.form.getlist('fruit'))

        bf_vege_input.append(bf_vege)
        bf_meat_input.append(bf_meat)
        bf_carb_input.append(bf_carb)
        bf_fruit_input.append(bf_fruit)

        lunch_vege_input = []
        lunch_meat_input = []
        lunch_carb_input = []
        lunch_fruit_input = []

        lunch_vege = random.choice(request.form.getlist('vege'))
        lunch_meat = random.choice(request.form.getlist('meat'))
        lunch_carb = random.choice(request.form.getlist('carb'))
        lunch_fruit = random.choice(request.form.getlist('fruit'))

        lunch_vege_input.append(lunch_vege)
        lunch_meat_input.append(lunch_meat)
        lunch_carb_input.append(lunch_carb)
        lunch_fruit_input.append(lunch_fruit)

        snack_vege_input = []
        snack_meat_input = []
        snack_carb_input = []
        snack_fruit_input = []

        snack_vege = random.choice(request.form.getlist('vege'))
        snack_meat = random.choice(request.form.getlist('meat'))
        snack_carb = random.choice(request.form.getlist('carb'))
        snack_fruit = random.choice(request.form.getlist('fruit'))

        snack_vege_input.append(snack_vege)
        snack_meat_input.append(snack_meat)
        snack_carb_input.append(snack_carb)
        snack_fruit_input.append(snack_fruit)

        dinner_vege_input = []
        dinner_meat_input = []
        dinner_carb_input = []
        dinner_fruit_input = []

        dinner_vege = random.choice(request.form.getlist('vege'))
        dinner_meat = random.choice(request.form.getlist('meat'))
        dinner_carb = random.choice(request.form.getlist('carb'))
        dinner_fruit = random.choice(request.form.getlist('fruit'))

        dinner_vege_input.append(dinner_vege)
        dinner_meat_input.append(dinner_meat)
        dinner_carb_input.append(dinner_carb)
        dinner_fruit_input.append(dinner_fruit)
        
        type_breakfast = request.form.getlist('breakfast_dishes')
        type_lunch =  request.form.getlist('lunch_dishes')
        type_dinner = request.form.getlist('dinner_dishes')
        type_snack = request.form.getlist('snack_dishes')
        print(type_breakfast)
        allergy_input = []
        vegan_input = []
        allergy_input.append(session['u_info'][13])
        vegan_input.append(session['u_info'][12])

        print(type_breakfast,allergy_input,vegan_input)
        time_breakfast = ['Breakfast']
        time_snack = ['Snack']
        time_lunch = ['Lunch']
        time_dinner = ['Dinner']

        def input_encode(entry, room):
            meal = dataset.values.tolist()
            meal_encode = dataset_encoded.values.tolist()
            lists = []
            encode = []
    
            for i in entry:
                found = False
                for j in meal:
                    if i == j[room]:
                        lists.append(j)
                        found = True
                        break
            if not found:
                print(f"Warning: '{i}' not found in dataset column {room}")
    
            for j in lists:
                encode.append(meal_encode[meal.index(j)][room])
        
            if not encode:  # No match found
                encode = [0]  # Set a default value or handle accordingly

            return encode
#         return meal_encode[meal.index(j)][room]

        def input_decode(entry,room): 
            meal = dataset.values.tolist()
            meal_encode = dataset_encoded.values.tolist()
            lists = []
            decode = []   
            for i in entry:
                for j in meal_encode:
                    if i==j[room]:
                        lists.append(j)
                        break
                 
            for j in lists: 
                decode.append(meal[meal_encode.index(j)][room])
        
            return decode

        bf_carb_encode = input_encode(bf_carb_input,1)[0]
        bf_meat_encode = input_encode(bf_meat_input,2)[0]
        bf_vege_encode = input_encode(bf_vege_input,3)[0]
        bf_fruit_encode = input_encode(bf_fruit_input,4)[0]

        lunch_carb_encode = input_encode(lunch_carb_input,1)[0]
        lunch_meat_encode = input_encode(lunch_meat_input,2)[0]
        lunch_vege_encode = input_encode(lunch_vege_input,3)[0]
        lunch_fruit_encode = input_encode(lunch_fruit_input,4)[0]

        dinner_carb_encode = input_encode(dinner_carb_input,1)[0]
        dinner_meat_encode = input_encode(dinner_meat_input,2)[0]
        dinner_vege_encode = input_encode(dinner_vege_input,3)[0]
        dinner_fruit_encode = input_encode(dinner_fruit_input,4)[0]

        snack_carb_encode = input_encode(snack_carb_input,1)[0]
        snack_meat_encode = input_encode(snack_meat_input,2)[0]
        snack_vege_encode = input_encode(snack_vege_input,3)[0]
        snack_fruit_encode = input_encode(snack_fruit_input,4)[0]
                
        breakfast_encode = input_encode(type_breakfast,5)[0]
        lunch_encode = input_encode(type_lunch,5)[0]
        snack_encode = input_encode(type_snack,5)[0]
        dinner_encode = input_encode(type_dinner,5)[0]

        vegan_encode = input_encode(vegan_input,6)[0]
        allergy_encode = input_encode(allergy_input,7)[0]
        bf_time_encode = input_encode(time_breakfast,8)[0]
        lunch_time_encode = input_encode(time_lunch,8)[0]
        snack_time_encode = input_encode(time_snack,8)[0]
        dinner_time_encode = input_encode(time_dinner,8)[0]

        bf_input = [bf_carb_encode,bf_meat_encode,bf_vege_encode,bf_fruit_encode,breakfast_encode,vegan_encode,allergy_encode,bf_time_encode]
        
        bf_result = model.predict([bf_input])
        bf_prediction = input_decode(bf_result,0)	
            
        lunch_input = [lunch_carb_encode,lunch_meat_encode,lunch_vege_encode,lunch_fruit_encode,lunch_encode,vegan_encode,allergy_encode,lunch_time_encode]
        lunch_result = model.predict([lunch_input])
        lunch_prediction = input_decode(lunch_result,0)	
            
        snack_input = [snack_encode,snack_encode,snack_encode,snack_encode,snack_encode,vegan_encode,allergy_encode,snack_time_encode]
        snack_result = model.predict([snack_input])
        snack_prediction = input_decode(snack_result,0)	
                        
        dinner_input = [dinner_encode,dinner_encode,dinner_encode,dinner_encode,dinner_encode,vegan_encode,allergy_encode,dinner_time_encode]
        dinner_result = model.predict([dinner_input])
        dinner_prediction = input_decode(dinner_result,0)	
        
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from user where u_id=?",(session['uid'],))
                data = cur.fetchone()
                calorie = data[17]
                protein = data[14]
                carb = data[15]
                fat = data[16]

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

        bf_cal = int(int(calorie)* 0.30)
        snack_cal = int(int(calorie)* 0.10)
        lunch_cal = int(int(calorie)* 0.35)
        dinner_cal = int(int(calorie)* 0.25)

        bf_protein = int(int(protein)* 0.30)
        snack_protein = int(int(protein)* 0.10)
        lunch_protein = int(int(protein)* 0.35)
        dinner_protein = int(int(protein)* 0.25)
        
        bf_carb = int(int(carb)* 0.30)
        snack_carb = int(int(carb)* 0.10)
        lunch_carb = int(int(carb)* 0.35)
        dinner_carb = int(int(carb)* 0.25)

        bf_fat = int(int(fat)* 0.30)
        snack_fat = int(int(fat)* 0.10)
        lunch_fat = int(int(fat)* 0.35)
        dinner_fat = int(int(fat)* 0.25)

        return render_template('recommendation.html',bf_prediction = bf_prediction[0],
                                                     lunch_prediction = lunch_prediction[0],
                                                     snack_prediction = snack_prediction[0],
                                                     dinner_prediction = dinner_prediction[0],
                                                     bf_cal = bf_cal,
                                                     snack_cal = snack_cal,
                                                     lunch_cal = lunch_cal,
                                                     dinner_cal = dinner_cal,
                                                     bf_protein = bf_protein,
                                                     snack_protein = snack_protein,
                                                     lunch_protein = lunch_protein,
                                                     dinner_protein = dinner_protein,
                                                     bf_carb = bf_carb,
                                                     snack_carb = snack_carb,
                                                     lunch_carb = lunch_carb,
                                                     dinner_carb = dinner_carb,
                                                     bf_fat = bf_fat,
                                                     snack_fat = snack_fat,
                                                     lunch_fat = lunch_fat,
                                                     dinner_fat = dinner_fat,

                                                     )




@app.route("/recommend_setup", methods = ['GET','POST'])
def recommend_setup():
    if request.method == 'GET':
        print(session['u_info'][12])
        return render_template('recommendsetup.html')

    else:
        
        return render_template('recommendsetup.html')

@app.route("/progress", methods = ['GET','POST'])
def progress():
    if request.method == 'GET':
        uid = session['uid']
        track_date = datetime.today().strftime ('%Y-%m-%d')
        days = []
        display_day = []
        weeks = []
        day_weight = []
        week_weight = []
        sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
        edate = datetime.strptime(track_date, '%Y-%m-%d').date()
        delta = edate - sdate     

        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            days.append(str(day))

        
        
        split_list = [days[x:x+7] for x in range(0, len(days), 7)]
        weeknum = split_list.index(split_list[-1])+1
        pw_date = split_list[-1][-1]
        
        pw_weight =[]
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from progress where u_id=? and p_date=?",(uid,track_date))
                data = cur.fetchone()
                if not data:
                    cur.execute("insert into progress (u_id,p_date,p_weight) values (?,?,?)",(session['u_info'][0],track_date,session['u_info'][6]))
                    conn.commit()

                cur.execute("select * from progress where u_id=? and p_date=?",(uid,track_date))
                data2 = cur.fetchone()	
                for i in data2:
                    pw_weight.append(data[2])
                    

                cur.execute("select * from progress_week where u_id=? and pw_num=?",(uid,weeknum))
                week_exist = cur.fetchone()

                if not week_exist:
                    
                        
                    cur.execute("insert into progress_week (u_id,pw_num,pw_weight) values (?,?,?)",(session['u_info'][0],weeknum,pw_weight[0]))
                    conn.commit()



                
                    
            cur.execute("select * from progress_week where u_id=?",(uid,))
            week_data = cur.fetchall()
            for i in week_data:
                weeks.append("Week"+str(i[1]))
                week_weight.append(i[2])
            for i in days:
                cur.execute("select * from progress where u_id=? and p_date=?",(uid,i))
                get_weight = cur.fetchall()
                for i in get_weight:
                    day_weight.append(i[2])

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from progress where u_id=?",(uid,))
                dates = cur.fetchall()
                for i in dates:
                    getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                    dates = getdate.strftime("%B-%d")
            
                    display_day.append(dates)

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

        
            

        return render_template('progress.html',days = display_day[-7:],weeks = weeks[-7:],d_weight = day_weight[-7:],w_weight = week_weight[-7:])
        

    else:
        return render_template('progress.html')

@app.route("/daily_detail", methods = ['GET','POST'])
def daily_detail():
    if request.method == 'GET':
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from tracking where u_id=? and track_date=?",(session['uid'],datetime.today().strftime ('%Y-%m-%d')))
                i = cur.fetchone()

                

                getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                date = getdate.strftime("%B-%d-%Y")
                breakfast = i[2]
                lunch = i[3]
                snack = i[4]
                dinner = i[5]
                calorie = session['u_info'][17]
                protein = i[7]
                carb = i[8]
                fat = i[9]
                consumed = i[6]
                deficit = round(Decimal(calorie - i[6]), 2)
                result = "Reduced "+ str(round(Decimal(consumed/3500),4))+"lb of bodyweight (in theory)"
                deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('daily_detail.html',date = date,
                                                   breakfast = breakfast,
                                                   lunch = lunch,
                                                   snack = snack,
                                                   dinner = dinner,
                                                   calorie = calorie,
                                                   protein = protein,
                                                   carb = carb,
                                                   fat = fat,
                                                   consumed = consumed,
                                                   result = result,
                                                   deficit = deficits)

    else:
        getdate = request.form['date']
        weight = ""
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from tracking where u_id=? and track_date=?",(session['uid'],getdate,))
                i = cur.fetchone()
                
                
                print(i[1])
                


                
                getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                date = getdate.strftime("%B-%d-%Y")
                breakfast = i[2]
                lunch = i[3]
                snack = i[4]
                dinner = i[5]
                calorie = int(session['u_info'][17])
                protein = i[7]
                carb = i[8]
                fat = i[9]
                consumed = i[6]
                deficit = round(Decimal(calorie - i[6]), 2)
                result = "Reduced "+ str(round(Decimal(consumed/3500),4))+"lb of bodyweight (in theory)"
                deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

                cur.execute("select * from progress where u_id=? and p_date=?",(session['uid'],getdate,))
                weights = cur.fetchone()
                if weights:
                    for i in weights:
                        weight = weights[2]
                else:
                    weight = "undefined"
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('daily_detail.html',date = date,
                                                   breakfast = breakfast,
                                                   lunch = lunch,
                                                   snack = snack,
                                                   dinner = dinner,
                                                   calorie = calorie,
                                                   protein = protein,
                                                   carb = carb,
                                                   fat = fat,
                                                   consumed = consumed,
                                                   result = result,
                                                   deficit = deficits,
                                                   weight = weight)

@app.route("/weekly_detail", methods=['GET', 'POST'])
def weekly_detail():

    def build_split_list():
        track_date = datetime.today().strftime('%Y-%m-%d')
        sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
        edate = datetime.strptime(track_date, '%Y-%m-%d').date()
        delta = edate - sdate
        days = [str(sdate + timedelta(days=i)) for i in range(delta.days + 1)]
        return [days[x:x + 7] for x in range(0, len(days), 7)]

    def get_week_stats(conn, weeknum, split_list):
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM progress_week WHERE u_id=? AND pw_num=?",
            (session['uid'], weeknum)
        )
        wow = cur.fetchone()
        if not wow:
            return None
        weight_of_week = wow[2]

        calories, proteins, fats, carbs = [], [], [], []

        for day in split_list[weeknum - 1]:
            cur.execute(
                "SELECT * FROM tracking WHERE u_id=? AND track_date=? AND track_calorie!=0",
                (session['uid'], day)
            )
            rows = cur.fetchall()
            for row in rows:
                calories.append(row[6])
                proteins.append(row[7])
                carbs.append(row[8])
                fats.append(row[9])

        if not calories:
            return None

        n = len(calories)
        calorie_consumed  = sum(calories)
        calorie_required  = float(session['u_info'][17]) * n
        deficit           = calorie_required - calorie_consumed

        return dict(
            average_calorie  = round(Decimal(sum(calories)  / n), 2),
            average_protein  = round(Decimal(sum(proteins)  / n), 2),
            average_carb     = round(Decimal(sum(carbs)     / n), 2),
            average_fat      = round(Decimal(sum(fats)      / n), 2),
            average_deficit  = round(Decimal(deficit        / n), 2),
            net_deficit      = round(Decimal(deficit),            2),
            loss_weight      = round(Decimal(deficit / 3500),     2),
            weight_of_week   = weight_of_week,
        )

    # ── main logic ──────────────────────────────────────────────────────────────
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # ✅ weeks list: one entry per week (for the dropdown)
            cur.execute(
                "SELECT pw_num FROM progress_week WHERE u_id=? ORDER BY pw_num ASC",
                (session['uid'],)
            )
            weeks = [row[0] for row in cur.fetchall()]

            split_list = build_split_list()

            if request.method == 'GET':
                this_weeknum = len(split_list)          # default = current week
            else:
                this_weeknum = int(request.form['weeks'])

            stats = get_week_stats(conn, this_weeknum, split_list)

    except sqlite3.Error as e:
        return str(e)

    if stats is None:
        return "No tracking data found for this week."

    result = f"Reduced {stats['loss_weight']}lb of bodyweight in this whole week (in theory)"

    return render_template(
        'weekly_detail.html',
        weeks          = weeks,
        week           = this_weeknum,
        weight_week    = stats['weight_of_week'],
        average_calorie= stats['average_calorie'],
        average_protein= stats['average_protein'],
        average_carb   = stats['average_carb'],
        average_fat    = stats['average_fat'],
        average_deficit= stats['average_deficit'],
        net_deficit    = stats['net_deficit'],
        result         = result,
    )


@app.route("/index", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM user WHERE u_id=?", (session['uid'],))
                u_data = cur.fetchone()
                if u_data is None:
                    return "User not found", 404
                weight = u_data[6]
        except sqlite3.Error as e:
            return f"Database Error: {e}", 500
        finally:
            conn.close()
        return render_template('index.html', weight=weight)

    else:  # POST
        try:
            getweight = float(request.form['weight'])
            if getweight <= 0:
                raise ValueError("Weight must be positive")
        except (ValueError, KeyError):
            return "Invalid weight value", 400

        uid = session['uid']

        try:
            with get_connection() as conn:
                cur = conn.cursor()

                cur.execute("SELECT * FROM user WHERE u_id=?", (uid,))
                u_info = cur.fetchone()
                if u_info is None:
                    return "User not found", 404

                age       = u_info[4]
                gender    = str(u_info[3]).strip().lower()   # safe string
                feet      = u_info[8]
                inches    = u_info[9]
                # Fix: safely convert activity_level to lowercase string
                activity_raw = u_info[11]
                if activity_raw is None:
                    activity_level = "sedentary"
                else:
                    # Convert to string and lower — handles int, float, str
                    activity_level = str(activity_raw).strip().lower()

                height_cm = (feet * 30.48) + (inches * 2.54)

                # BMR Mifflin-St Jeor
                if gender == "male":
                    bmr = 10 * getweight + 6.25 * height_cm - 5 * age + 5
                else:
                    bmr = 10 * getweight + 6.25 * height_cm - 5 * age - 161

                # Activity map (all keys lowercase)
                act_map = {
                    "sedentary": 1.2,
                    "lightly active": 1.375,
                    "moderately active": 1.55,
                    "very active": 1.725,
                    "extra active": 1.9
                }

                # Safe lookup
                multiplier = act_map.get(activity_level, 1.2)
                maintenance_calories = int(bmr * multiplier)

                # Deficit + safety floor
                target_calories = maintenance_calories - 500

                MIN_CALORIES_MALE   = 1600
                MIN_CALORIES_FEMALE = 1400
                min_calories = MIN_CALORIES_MALE if gender == "male" else MIN_CALORIES_FEMALE

                if target_calories < min_calories:
                    target_calories = min_calories

                # Min protein 1.2 g/kg (good for 60+)
                min_protein_g = max(60, int(getweight * 1.2))

                # Macros: higher protein emphasis
                protein = int((target_calories * 0.35) / 4)
                carb    = int((target_calories * 0.40) / 4)
                fat     = int((target_calories * 0.25) / 9)

                if protein < min_protein_g:
                    protein = min_protein_g

                # Update user
                cur.execute("""
                    UPDATE user SET 
                        u_weight = ?,
                        u_calories = ?,
                        u_protein = ?,
                        u_carb = ?,
                        u_fat = ?
                    WHERE u_id = ?""",
                    (getweight, target_calories, protein, carb, fat, uid))

                # Update today's progress
                today = date.today().isoformat()
                cur.execute("""
                    UPDATE progress SET p_weight = ?
                    WHERE p_date = ? AND u_id = ?""",
                    (getweight, today, uid))

                conn.commit()

                # Refresh session
                cur.execute("SELECT * FROM user WHERE u_id=?", (uid,))
                session['u_info'] = list(cur.fetchone())

                # Weekly average logic (simplified + safer)
                try:
                    start_date_str = u_info[-1]  # assuming last column = join/start date
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                except:
                    start_date = date.today()

                current_date = date.today()
                delta_days = (current_date - start_date).days

                if delta_days >= 0:
                    all_days = [start_date + timedelta(days=i) for i in range(delta_days + 1)]
                    weeks = [all_days[i:i+7] for i in range(0, len(all_days), 7)]
                    current_week_days = weeks[-1]

                    week_weights = []
                    for d in current_week_days:
                        cur.execute(
                            "SELECT p_weight FROM progress WHERE u_id=? AND p_date=?",
                            (uid, d.isoformat())
                        )
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            week_weights.append(row[0])

                    if week_weights:
                        avg_weight = round(sum(week_weights) / len(week_weights), 2)
                        this_week_num = len(weeks)

                        cur.execute("""
                            INSERT OR REPLACE INTO progress_week (pw_num, u_id, pw_weight)
                            VALUES (?, ?, ?)""",
                            (this_week_num, uid, avg_weight))
                        conn.commit()

        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            return f"Error: {str(e)}", 500
        finally:
            if 'conn' in locals():
                conn.close()

        return redirect(url_for('index'))

@app.route("/about", methods = ['GET','POST'])
def about():
    if request.method == 'GET':
        return render_template('about.html')

    else:
        return render_template('about.html')

@app.route('/logout')
def logout():
    
    session.pop('uid',None)
    session.pop('u_pass',None)
    session.pop('u_info',None)
    session.pop('bf_meal',None)
    session.pop('bf_numbers',None)
    session.pop('lunch_meal',None)
    session.pop('lunch_numbers',None)
    session.pop('dinner_meal',None)
    session.pop('dinner_numbers',None)
    session.pop('snack_meal',None)
    session.pop('snack_numbers',None)
    bf_meal.clear()
    lunch_meal.clear()
    snack_meal.clear()
    dinner_meal.clear()
    flash('You have successfully logged out')
    return redirect(url_for('login'))
# ================= WESTERN FOOD ROUTE ================= #

@app.route('/western_predict', methods=['POST'])
def western_predict():

    if 'file' not in request.files:
        return render_template('index1.html', error="No file selected")

    file = request.files['file']

    if file.filename == '':
        return render_template('index1.html', error="No file selected")

    if file and allowed_file(file.filename):

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Save temp image
        temp_path = "temp_western.jpg"
        cv2.imwrite(temp_path, img)

        # Western YOLO prediction
        results = western_model.predict(temp_path, verbose=False)

        class_id = results[0].probs.top1
        food_name = results[0].names[class_id]
        confidence = float(results[0].probs.top1conf)
        calories = yolo_food_calories.get(food_name, "Unknown")

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode()

        return render_template(
    'index1.html',
    filename=f'data:image/jpg;base64,{image_base64}',
    food=food_name,
    confidence=round(confidence * 100, 2),
    calories=calories,
    items_with_calories={},
    total_calories=calories,
    western=True   # 👈 ADD THIS
    )

    return render_template('index1.html', error="Invalid file format")
if __name__=="__main__":
    app.run(port=5000,debug="true")
