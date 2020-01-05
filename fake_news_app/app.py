from chalice import Chalice, Response
from jinja2 import Environment, FileSystemLoader
import urllib
import requests

app = Chalice(app_name='fake_news_app')
loader = FileSystemLoader('chalicelib/templates')

@app.route('/')
def index():
    view = Environment(loader=loader).get_template('index.html').render(failed=False)
    return Response(view, status_code=200, 
                    headers={"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"})

@app.route('/relay', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def send_article():
    text = urllib.parse.parse_qs(app.current_request.__dict__['_body'])[b'article'][0]
    data = requests.post(url='http://localhost:8080/invocations', data={'article': text.decode('utf-8')})
    print(data.text)
    data = float(data.text)
    if data < 0:
        view = Environment(loader=loader).get_template('index.html').render(failed=True)
    else:
        view = Environment(loader=loader).get_template('response.html').render(score=data)
    return Response(view, status_code=200, 
                    headers={'Content-Type': 'text/html', 'Access-Control-Allow-Origin': '*'})