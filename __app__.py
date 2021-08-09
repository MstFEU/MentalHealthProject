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
      
      
      
   


#answers = []

#@app.route("/", methods=["GET", "POST"])
#def home():
#    if request.method == "POST":
#        print(request.form.get("self-employed"))
#        #คำสั่งเก็บข้อมูลเป็น tuple แล้วเพิ่มเข้า tansanctions list
#        answers.append(
#            (
#                int(request.form.get("self-employed")),
#                int(request.form.get("no-of-employees")),
#                int(request.form.get("tech-company")),
#                int(request.form.get("mental-healthcare-coverage")),
#                int(request.form.get("knowledge-about-mental-healthcare-options-workplace")),
#                int(request.form.get("employer-discussed-mental-health")),
#                int(request.form.get("employer-offer-resources-to-learn-about-mental-health")),
#                int(request.form.get("medical-leave-from-work")),
#                int(request.form.get("comfortable-discussing-with-coworkers")),
#                int(request.form.get("employer-take-mental-health-seriously")),
#                int(request.form.get("openess-of-family-friends")),
#                int(request.form.get("family-history-mental-illness")),
#                int(request.form.get("mental-health-disorder-past")),
#                int(request.form.get("currently-mental-health-disorder")),
#                int(request.form.get("age")),
#                int(request.form.get("gender")),
#                int(request.form.get("work-remotely")),
#                int(request.form.get("tech-role"))
#                
#            )
#        )
#    print(answers) 
#    return render_template("question.html")



#@app.route('/ressult')
#def result():
      
#   return render_template('result.html')
   
#to question page and predict
#@app.route('/question', methods=["POST"])
#def predict():
#      a = int(request.form["self-employed"]),
#      b = int(request.form["no-of-employees"]),
#      c = int(request.form["tech-company"]),
#      d = int(request.form["mental-healthcare-coverage"]),
#      e = int(request.form["knowledge-about-mental-healthcare-options-workplace"]),
#      f = int(request.form["employer-discussed-mental-health"]),
#      g = int(request.form["employer-offer-resources-to-learn-about-mental-health"]),
#      h = int(request.form["medical-leave-from-work"]),
#      i = int(request.form["comfortable-discussing-with-coworkers"]),
#      j = int(request.form["employer-take-mental-health-seriously"]),
#      k = int(request.form["openess-of-family-friends"]),
#      l = int(request.form["family-history-mental-illness"]),
#      m = int(request.form["mental-health-disorder-past"]),
#      n = int(request.form["currently-mental-health-disorder"]),
#      o = int(request.form["age"]),
#      p = int(request.form["gender"]),
#      q = int(request.form["work-remotely"]),
#      r = int(request.form["tech-role"])
#      list = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]])
#      model = pickle.load(open("model.pkl", "rb"))
#      result = model.predict(list)
#      return render_template('question.html')


#@app.route("/question", methods=["GET", "POST"])
#def predictor():
   #if request.method == "POST":
      
      #model = pickle.load(open('model.pkl', 'rb'))
      #result = model.predict(answers)
      #return result[0]
   #print(answers)
   #return render_template('question.html')

#@app.route('/result', methods=['POST'])
#def result():
   #if request.method == 'POST':
     # answers = request.form.to_dict()
     # answers = list(answers.values())
     # answers = list(map(int, answers))
     # result = ValuePredictor(answers)
     # if int(result)=="Yes":
     #    prediction = "YES"
     # else: 
     #    prediction = "NO"
     # return render_template("result.html", prediction = prediction)

if __name__=='__main__':
   app.run()

   


