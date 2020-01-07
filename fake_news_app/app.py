from chalice import Chalice, Response
from jinja2 import Environment, FileSystemLoader
import requests
import urllib.parse

app = Chalice(app_name='fake_news_app')
loader = FileSystemLoader('chalicelib/templates')

@app.route('/')
def index():
    view = Environment(loader=loader).get_template('index.html').render(failed=False)
    return Response(view, status_code=200, 
                    headers={"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"})

@app.route('/relay', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def send_article():
    article = app.current_request.raw_body.decode('utf-8').split('=', 1)[1]
    print(article, '\n\n')
    article = urllib.parse.unquote_plus(article)
    print(article)
    #print(f'{urllib.parse.unquote_plus(article)}')

    sagemaker_response = requests.post(url='http://localhost:8080/invocations', data={'article': article})
    data = float(sagemaker_response.text)
    if data < 0:
        view = Environment(loader=loader).get_template('index.html').render(failed=True)
    else:
        view = Environment(loader=loader).get_template('response.html').render(score=data)
    return Response(view, status_code=200, 
                    headers={'Content-Type': 'text/html', 'Access-Control-Allow-Origin': '*'})