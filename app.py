from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the models
models = {}
pickle_in = open('model_fakenews1.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tfid1.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)
# Load the Random Forest model



with open('model_randomforest.pickle', 'rb') as f:
    models['Random Forest'] = pickle.load(f)

# Load the XGBoost model
with open('model_xgboost.pickle', 'rb') as f:
    models['XGBoost'] = pickle.load(f)

# Load the SVM model
with open('model_svm.pickle', 'rb') as f:
    models['SVM'] = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfid1.pickle', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/newscheck')
def newscheck():    
    abc = request.args.get('news')  
    input_data = [abc.rstrip()]
    # transforming input
    tfidf_test = tfidf_vectorizer.transform(input_data)
    # predicting the input
    y_pred = pac.predict(tfidf_test)
    print(y_pred)
    return jsonify(result = y_pred[0])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
