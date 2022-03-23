from flask import Flask, render_template, request, url_for, redirect
from data_model import ProductRecommendation

product_recommendation = ProductRecommendation()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        flag = request.args.get('flag', False)
        return render_template('homepage.html', flag=flag)
    elif request.method == "POST":
        name = request.form['Username']
        return redirect(url_for('submit', name=name))


@app.route('/submit/<name>', methods=['GET', 'POST'])
def submit(name):
    if product_recommendation.valid_input_checker(name):
        df = product_recommendation.get_recommendation_from_username(name)
        return render_template('result.html', name=name, tables=[df.to_html(classes='data', header=True)])
    else:
        flag = True
        return redirect(url_for('home', name=name, flag=flag))


if __name__ == '__main__':
    app.run()
