from chalice import Chalice, Response
from jinja2 import Environment, FileSystemLoader

app = Chalice(app_name='fake_news_app')
loader = FileSystemLoader('chalicelib/templates')

@app.route('/')
def index():
    view = Environment(loader=loader).get_template('index.html').render()
    return Response(view, status_code=200, 
                    headers={"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"})


# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
