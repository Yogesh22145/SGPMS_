import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from io import BytesIO
import base64
import pandas as pd
import numpy as np

app = Flask(__name__)

df = pd.read_csv("DPM.csv")
df = df.drop(['DPM'], axis = 1)
df = df.rename(columns=df.iloc[0])
df = df.drop([0,1,2,3])
for col in list(df.columns):
    df[col] = np.array(df[col], dtype=float)
counts, bins = np.histogram(df['Total(Scaled)'], bins = 9)
cgpa = []
for row in df['Total(Scaled)']:
  if row > bins[-2]:
    cgpa.append(10)
  elif row > bins[-3]:
    cgpa.append(10)
  elif row > bins[-4]:
    cgpa.append(9)
  elif row > bins[-5]:
    cgpa.append(8)
  elif row > bins[-6]:
    cgpa.append(7)
  elif row > bins[-7]:
    cgpa.append(6)
  elif row > bins[-8]:
    cgpa.append(5)
  elif row > bins[-9]:
    cgpa.append(4)
  else:
    cgpa.append(2)

df['CGPA'] = cgpa

new_df = df.drop(['Total(Scaled)'], axis = 1)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_df[new_df.columns[:-1]], new_df["CGPA"], test_size=0.2, random_state=42)

scalar.fit_transform(X_train)
scalar.fit(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeRegressor

def train(x,y,lpt):
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  regressor = DecisionTreeRegressor()
  regressor.fit(X_train,y_train)
  px = regressor.predict([lpt])
  return px[0]

#home page
@app.route('/')
def index():
    return render_template('HomePage.html')

@app.route('/first', methods=['GET', 'POST'])
def first():
    if request.method == "POST":
        course_inp = request.form.get("course_Input")  # inp - getting Input from Homepage
        print("Selected course : "+course_inp)
    return render_template('FirstPage.html')

@app.route('/second', methods=['GET', 'POST'])
def second():
    if request.method == "POST":
        pass
    return render_template('SecondPage.html')

@app.route('/third', methods=['GET', 'POST'])
def third():
    if request.method == "POST":
        pass
    return render_template('ThirdPage.html')

@app.route('/predict_efforts', methods=['GET', 'POST'])
def predict_efforts():
    if request.method == "POST":
        cgpa = request.form.get("CGPA")
        assignment1 = request.form.get("assignment1")
        assignment2 = request.form.get("assignment2")
        assignment3 = request.form.get("assignment3")
        assignment4 = request.form.get("assignment4")
        project = request.form.get("project")
        mid_sem = request.form.get("mid_sem")
        end_sem = request.form.get("end_sem")
        print("Assignment 1 Efforts : " + assignment1)
        print("Assignment 2 Efforts : " + assignment2)
        print("Assignment 3 Efforts : " + assignment3)
        print("Assignment 4 Efforts : " + assignment4)
        print("Project Efforts : " + project)
        print("Mid Sem Efforts : " + mid_sem)
        print("End Sem Efforts : " + end_sem)
        print("CGPA Efforts : " + cgpa)

        lpt = []
        inp = []
        try:
            if assignment1 == "":
                inp.append("Assignment 1")
            elif float(assignment1) < 0 or float(assignment1) > 100:
                return "<h1>Please Enter Valid Assignment 1 Marks [0-100]</h1>"
            else:
                lpt.append(float(assignment1))

            if assignment2 == "":
                inp.append("Assignment 2")
            elif float(assignment2) < 0 or float(assignment2) > 100:
                return "<h1>Please Enter Valid Assignment 2 Marks [0-100]</h1>"
            else:
                lpt.append(float(assignment2))

            if mid_sem == "":
                inp.append("Midsem")
            elif float(mid_sem) < 0 or float(mid_sem) > 40:
                return "<h1>Please Enter Valid Mid Sem Marks [0-40]</h1>"
            else:
                lpt.append(float(mid_sem))

            if assignment3 == "":
                inp.append("Assignment 3")
            elif float(assignment3) < 0 or float(assignment3) > 100:
                return "<h1>Please Enter Valid Assignment 3 Marks [0-100]</h1>"
            else:
                lpt.append(float(assignment3))

            if assignment4 == "":
                inp.append("Assignment 4")
            elif float(assignment4) < 0 or float(assignment4) > 100:
                return "<h1>Please Enter Valid Assignment 4 Marks [0-100]</h1>"
            else:
                lpt.append(float(assignment4))

            if end_sem == "":
                inp.append("Endsem Exam")
            elif float(end_sem) < 0 or float(end_sem) > 100:
                return "<h1>Please Enter Valid End Sem Marks [0-100]</h1>"
            else:
                lpt.append(float(end_sem))

            if project == "":
                inp.append("Project")
            elif float(project) < 0 or float(project) > 100:
                return "<h1>Please Enter Valid Project Marks [0-100]</h1>"
            else:
                lpt.append(float(project))

            if cgpa == "":
                return "<h1>Please Enter your Expected CGPA</h1>"
            elif float(cgpa) not in [2,4,5,6,7,8,9,10]:
                return "<h1>Please Enter Valid CGPA [2,4,5,6,7,8,9,10]</h1>"
            else:
                lpt.append(float(cgpa))
        except:
            return "<h1>Please Enter Valid Marks (may be entered characters instead of numericals)</h1>"

        if len(inp) >= 7:
            return "<h1>Please Enter at least one criteria</h1>"
        elif len(inp) == 0:
            return "<h1>All efforts are entered, requires no prediction</h1>"

        px2 = list(new_df.columns)
        for item in inp:
            if item in px2:
                px2.remove(item)

        lst = []
        x = new_df[px2].values

        print(px2)
        print(lpt)

        for col in inp:
            y = new_df[col].values
            lst.append(train(x, y, lpt))

        df2 = pd.DataFrame([lst], columns=inp)

    return render_template('ThirdPage.html', tables=[df2.to_html(classes='data', index = False)], titles=df2.columns.values)

@app.route('/predict_results', methods=['GET', 'POST'])
def predict_results():
    if request.method == "POST":
        assignment1 = request.form.get("assignment1")
        assignment2 = request.form.get("assignment2")
        assignment3 = request.form.get("assignment3")
        assignment4 = request.form.get("assignment4")
        project = request.form.get("project")
        mid_sem = request.form.get("mid_sem")
        end_sem = request.form.get("end_sem")
        print("Assignment 1 Efforts : " + assignment1)
        print("Assignment 2 Efforts : " + assignment2)
        print("Assignment 3 Efforts : " + assignment3)
        print("Assignment 4 Efforts : " + assignment4)
        print("Project Efforts : " + project)
        print("Mid Sem Efforts : " + mid_sem)
        print("End Sem Efforts : " + end_sem)

        try:
            if assignment1 == "":
                return "<h1>Please Enter Assignment 1 Marks [0-100]</h1>"
            elif float(assignment1) < 0 or float(assignment1) > 100:
                return "<h1>Please Enter Valid Assignment 1 Marks [0-100]</h1>"
            else:
                pass

            if assignment2 == "":
                return "<h1>Please Enter Assignment 2 Marks [0-100]</h1>"
            elif float(assignment2) < 0 or float(assignment2) > 100:
                return "<h1>Please Enter Valid Assignment 2 Marks [0-100]</h1>"
            else:
                pass

            if mid_sem == "":
                return "<h1>Please Enter Mid Sem Marks [0-40]</h1>"
            elif float(mid_sem) < 0 or float(mid_sem) > 40:
                return "<h1>Please Enter Valid Mid Sem Marks [0-40]</h1>"
            else:
                pass

            if assignment3 == "":
                return "<h1>Please Enter Assignment 3 Marks [0-100]</h1>"
            elif float(assignment3) < 0 or float(assignment3) > 100:
                return "<h1>Please Enter Valid Assignment 3 Marks [0-100]</h1>"
            else:
                pass

            if assignment4 == "":
                return "<h1>Please Enter Assignment 4 Marks [0-100]</h1>"
            elif float(assignment4) < 0 or float(assignment4) > 100:
                return "<h1>Please Enter Valid Assignment 4 Marks [0-100]</h1>"
            else:
                pass

            if end_sem == "":
                return "<h1>Please Enter End Sem Marks [0-100]</h1>"
            elif float(end_sem) < 0 or float(end_sem) > 100:
                return "<h1>Please Enter Valid End Sem Marks [0-100]</h1>"
            else:
                pass

            if project == "":
                return "<h1>Please Enter Project Marks [0-100]</h1>"
            elif float(project) < 0 or float(project) > 100:
                return "<h1>Please Enter Valid Project Marks [0-100]</h1>"
            else:
                pass
        except:
            return "<h1>Please Enter Valid Marks (may be entered characters instead of numericals)</h1>"

        res = classifier.predict([[float(assignment1), float(assignment2), float(mid_sem), float(assignment3),
                                   float(assignment4), float(end_sem), float(project)]])
        return "<h1>The Predicted CGPA is "+str(res)+"</h1>"
    return render_template('FrontEnd.html')

@app.route('/shiva', methods=['GET', 'POST'])
def front():
    if request.method == "POST":
        pass
    return render_template('FrontEnd.html')

@app.route('/yogesh', methods=['GET', 'POST'])
def plot():
    if request.method == "POST":
        img = BytesIO()
        inp = request.form.get("Input")
        print("Selected Task : "+inp)
        plt.figure(figsize=(12,6))
        (n, bins, patches) = plt.hist(df[inp], bins = 9, edgecolor = 'black')
        plt.xticks(bins)
        plt.xlabel('Marks')
        plt.ylabel('Frequency')
        plt.title(inp)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('FrontEnd.html', plot_url=plot_url)


if __name__ == '__main__':
    app.run()
