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
def hello_name(button):
    rm = app.config['RUNMANAGER']
    buttonInt = int(button)
    note = rm.next(buttonInt)
    noteString = str(note)
    print(buttonInt, noteString)
    return noteString

def run_app(rm):
    app.config['RUNMANAGER'] = rm
    app.run(host="10.38.48.112")
