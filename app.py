import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import numpy as np
# import the necessary packages
from capture_image import take_new_user_picture
from facial_recognition_model import run_facial_recognition_Model
from facial_recognition_system import run_facial_recognition_system
from run_attendance_system import run_facial_recognition_attendance_system

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
# cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('dataset'):
    os.makedirs('dataset')
if not os.path.isdir('Users'):
    os.makedirs('Users')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,ID,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('dataset'))

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    ids = df['ID']
    times = df['Time']
    l = len(df)
    return names,ids,times,l


# add user in user list
def add_user(name, id, type):
    df = pd.read_csv(f'Users/userlist.csv')
    if id not in list(df['ID']):
        with open(f'Users/userlist.csv','a') as f:
            f.write(f'\n{name},{id},{type}')

# app routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        new_user_name = request.form.get('username')
        new_user_id = request.form.get('userid')
        new_user_type = request.form.get('users')
        userimagefolder = 'dataset/'+new_user_id
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        take_new_user_picture(new_user_id)

        if f'userlist.csv' not in os.listdir('Users'):
            with open(f'Users/userlist.csv','w') as f:
                f.write('Name,ID,Type')
        add_user(new_user_name, new_user_id, new_user_type)
    
    return render_template('register.html', totalreg=totalreg())

@app.route('/train', methods=['GET'])
def train():
    run_facial_recognition_Model()
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def start():
    run_facial_recognition_system()
    return render_template('index.html')

@app.route('/attendance', methods=['GET'])
def attendance():
    names,ids,times,l = extract_attendance()
    return render_template('attendance.html', names=names,ids=ids,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

@app.route('/take', methods=['GET'])
def take():
    run_facial_recognition_attendance_system()
    names,ids,times,l = extract_attendance()
    return render_template('attendance.html', names=names,ids=ids,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)




#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)