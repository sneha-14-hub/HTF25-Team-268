from flask import Flask, render_template, request
from check import predict_text, predict_url

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_type = request.form.get("input_type")
    result = ""
    user_input = None  # to send back user input for display if needed

    if input_type == "text":
        news_text = request.form.get("news_text")
        user_input = news_text
        if news_text:
            result = predict_text(news_text)
        else:
            result = "Please enter article text."

    elif input_type == "url":
        news_url = request.form.get("news_url")
        user_input = news_url
        if news_url:
            result = predict_url(news_url)
        else:
            result = "Please enter a news URL."

    else:
        result = "Invalid input type."

    return render_template("result.html", result=result, user_url=user_input)

if __name__ == "__main__":
    app.run(debug=True)