import numpy as np
import pickle
from flask import Flask, render_template, request

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

#to homepage
@app.route('/')
def home():
   return render_template('home.html')

#to info on mental health
@app.route('/info')
def info():
   return render_template('info.html')

#to what is the mentalist
@app.route('/mentalist')
def mentalist():
   return render_template('mentalist.html')

#to about us
@app.route('/us')
def us():
   return render_template('us.html')

@app.route('/question', methods=["GET", "POST"])
def question(): 
      if request.method == "POST":
         a = int(request.form["self-employed"])
         b = int(request.form["no-of-employees"])
         c = int(request.form["tech-company"])
         d = int(request.form["mental-healthcare-coverage"])
         e = int(request.form["knowledge-about-mental-healthcare-options-workplace"])
         f = int(request.form["employer-discussed-mental-health"])
         g = int(request.form["employer-offer-resources-to-learn-about-mental-health"])
         h = int(request.form["medical-leave-from-work"])
         i = int(request.form["comfortable-discussing-with-coworkers"])
         j = int(request.form["employer-take-mental-health-seriously"])
         k = int(request.form["openess-of-family-friends"])
         l = int(request.form["family-history-mental-illness"])
         m = int(request.form["mental-health-disorder-past"])
         n = int(request.form["currently-mental-health-disorder"])
         o = int(request.form["age"])
         p = int(request.form["gender"])
         q = int(request.form["work-remotely"])
         r = int(request.form["tech-role"])
         answer = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]])
         print(answer)
         pred = model.predict(answer)
         print(pred)
         return render_template("result.html", data = pred)
      else: 
         return render_template("question.html") 

if __name__=='__main__':
   app.run()

   


