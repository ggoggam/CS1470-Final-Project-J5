from flask import Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def parse_request():
    print(request.form.get('name'))
    rm = app.config['RUNMANAGER']
    note = rm.next(3)
    print(note)
    return "hi"

@app.route('/<button>')
def hello_name(name):
    return "Hello {}!".format(name)

def run_app(rm):
    app.config['RUNMANAGER'] = rm
    app.run()
