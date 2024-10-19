from app import create_app, db

app = create_app()  # Create the app instance

@app.shell_context_processor
def make_shell_context():
    return {'db': db}

if __name__ == '__main__':
    app.run(port=7237, debug=True)